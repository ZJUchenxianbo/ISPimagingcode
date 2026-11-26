#include <algorithm>
#include <cstdio>
#include <memory>
#include <new>
#include <vector>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <cusparse.h>

#ifdef PANEL_DEVICE_ASSEMBLY
extern "C" cudaError_t launch_panel_assembly(cudaStream_t, int, const double *, const double *, double, double,
                                              int, const int *, const int *, const int *, const int *, int, int,
                                              const cuDoubleComplex *, cuDoubleComplex *);
#endif

namespace {

using Complex = cuDoubleComplex;

struct ResidualWorkspace {
    cublasHandle_t blas = nullptr;
    cusparseHandle_t sparse = nullptr;
    cusparseSpMatDescr_t upper = nullptr;
    cusparseDnVecDescr_t x = nullptr, ax = nullptr, values = nullptr;
    cusparseSpMatDescr_t coupling = nullptr;
    cusparseDnVecDescr_t panel_values = nullptr, panel_rhs_values = nullptr;
    cusparseSpVecDescr_t diagonal = nullptr;
    int *diagonal_positions = nullptr;
    Complex *diagonal_values = nullptr, *applied = nullptr, *diagonal_product = nullptr;
    Complex *panel_rhs = nullptr;
    void *scratch = nullptr, *panel_scratch = nullptr;

    ~ResidualWorkspace() {
        if (panel_rhs_values) cusparseDestroyDnVec(panel_rhs_values);
        if (panel_values) cusparseDestroyDnVec(panel_values);
        if (coupling) cusparseDestroySpMat(coupling);
        if (diagonal) cusparseDestroySpVec(diagonal);
        if (values) cusparseDestroyDnVec(values);
        if (ax) cusparseDestroyDnVec(ax);
        if (x) cusparseDestroyDnVec(x);
        if (upper) cusparseDestroySpMat(upper);
        if (sparse) cusparseDestroy(sparse);
        if (blas) cublasDestroy(blas);
        cudaFree(scratch);
        cudaFree(panel_scratch);
        cudaFree(panel_rhs);
        cudaFree(diagonal_positions);
        cudaFree(diagonal_product);
        cudaFree(applied);
        cudaFree(diagonal_values);
    }
};

struct BatchWorkspace {
    int count = 0;
    cudssMatrix_t rhs = nullptr, solution = nullptr;
    cusparseDnMatDescr_t values = nullptr, applied = nullptr, normal = nullptr;
    cublasHandle_t norm_blas = nullptr;
    Complex *rhs_values = nullptr, *solution_values = nullptr;
    Complex *original_rhs = nullptr, *applied_values = nullptr, *normal_values = nullptr;
    Complex *diagonal_product = nullptr;
    double *norms = nullptr;
    void *scratch = nullptr;
    std::vector<double> host_norms;

    ~BatchWorkspace() {
        if (norm_blas) cublasDestroy(norm_blas);
        if (normal) cusparseDestroyDnMat(normal);
        if (applied) cusparseDestroyDnMat(applied);
        if (values) cusparseDestroyDnMat(values);
        if (solution) cudssMatrixDestroy(solution);
        if (rhs) cudssMatrixDestroy(rhs);
        cudaFree(scratch);
        cudaFree(norms);
        cudaFree(diagonal_product);
        cudaFree(normal_values);
        cudaFree(applied_values);
        cudaFree(original_rhs);
        cudaFree(solution_values);
        cudaFree(rhs_values);
    }
};

#ifdef PANEL_DEVICE_ASSEMBLY
struct AssemblyWorkspace {
    int groups = 0, degree = 0, coefficients = 0, levels = 0;
    int *targets = nullptr, *offsets = nullptr, *sources = nullptr, *rows = nullptr;
    double *stiffness = nullptr, *mass = nullptr;
    Complex *pml = nullptr;

    ~AssemblyWorkspace() {
        cudaFree(pml);
        cudaFree(mass);
        cudaFree(stiffness);
        cudaFree(rows);
        cudaFree(sources);
        cudaFree(offsets);
        cudaFree(targets);
    }
};
#endif

struct Workspace {
    int device = 0, ice = 0, panels = 0;
    int head_nnz = 0, tail_nnz = 0;
    cudaStream_t stream = nullptr;
    cudssHandle_t solver = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr, rhs = nullptr, solution = nullptr, schur = nullptr;
    int *row = nullptr, *column = nullptr;
    Complex *values = nullptr, *rhs_values = nullptr, *solution_values = nullptr, *schur_values = nullptr;
    cudaEvent_t events[5]{};
    std::unique_ptr<ResidualWorkspace> residual;
    std::unique_ptr<BatchWorkspace> batch;
#ifdef PANEL_DEVICE_ASSEMBLY
    std::unique_ptr<AssemblyWorkspace> assembly;
#endif
    bool factored = false;
    bool solved = false;
    bool panel_solved = false;

    ~Workspace() {
        if (cudaSetDevice(device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);
        batch.reset();
        residual.reset();
#ifdef PANEL_DEVICE_ASSEMBLY
        assembly.reset();
#endif
        if (schur) cudssMatrixDestroy(schur);
        if (solution) cudssMatrixDestroy(solution);
        if (rhs) cudssMatrixDestroy(rhs);
        if (matrix) cudssMatrixDestroy(matrix);
        if (data) cudssDataDestroy(solver, data);
        if (config) cudssConfigDestroy(config);
        if (solver) cudssDestroy(solver);
        cudaFree(schur_values);
        cudaFree(solution_values);
        cudaFree(rhs_values);
        cudaFree(values);
        cudaFree(column);
        cudaFree(row);
        for (auto event : events) {
            if (event) cudaEventDestroy(event);
        }
        if (stream) cudaStreamDestroy(stream);
    }
};

#define CUDA(call) do { const auto rc = (call); if (rc != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(rc)); return int(rc); } } while (0)
#define DSS(call) do { const auto rc = (call); if (rc != CUDSS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s: cuDSS status %d\n", #call, int(rc)); return 1000 + int(rc); } } while (0)
#define SPARSE(call) do { const auto rc = (call); if (rc != CUSPARSE_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s: cuSPARSE status %d\n", #call, int(rc)); return 3000 + int(rc); } } while (0)
#define BLAS(call) do { const auto rc = (call); if (rc != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s: cuBLAS status %d\n", #call, int(rc)); return 4000 + int(rc); } } while (0)

constexpr auto sparse_algorithm = CUSPARSE_SPMV_CSR_ALG2;

int check_factor(Workspace &work) {
    size_t used = 0;
    int info = 0;
    DSS(cudssDataGet(work.solver, work.data, CUDSS_DATA_INFO, &info, sizeof(info), &used));
    if (info) std::fprintf(stderr, "Panel Schur cuDSS numeric status %d\n", info);
    return info ? 2001 : 0;
}

int factor_and_extract(Workspace &w, Complex *schur, double *times) {
    if (times) CUDA(cudaEventRecord(w.events[1], w.stream));
    DSS(cudssExecute(w.solver, CUDSS_PHASE_FACTORIZATION, w.config, w.data, w.matrix, w.solution, w.rhs));
    if (times) CUDA(cudaEventRecord(w.events[2], w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    if (const int status = check_factor(w)) return status;
    size_t used = 0;
    DSS(cudssDataGet(w.solver, w.data, CUDSS_DATA_SCHUR_MATRIX, &w.schur, sizeof(w.schur), &used));
    if (times) CUDA(cudaEventRecord(w.events[3], w.stream));
    CUDA(cudaMemcpyAsync(schur, w.schur_values, sizeof(Complex) * size_t(w.panels) * w.panels,
                         cudaMemcpyDeviceToHost, w.stream));
    if (times) CUDA(cudaEventRecord(w.events[4], w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    if (times) {
        for (int i = 0; i < 4; ++i) {
            float milliseconds = 0;
            CUDA(cudaEventElapsedTime(&milliseconds, w.events[i], w.events[i + 1]));
            times[i] = milliseconds * 0.001;
        }
    }
    w.factored = true;
    return 0;
}

#ifdef PANEL_DEVICE_ASSEMBLY
template <class T>
int copy_static(T *&device, const T *host, size_t count, cudaStream_t stream) {
    if (!count) return 0;
    CUDA(cudaMalloc(&device, sizeof(T) * count));
    CUDA(cudaMemcpyAsync(device, host, sizeof(T) * count, cudaMemcpyHostToDevice, stream));
    return 0;
}
#endif

int create_residual(Workspace &work, const int *positions) {
    std::unique_ptr<ResidualWorkspace> storage(new (std::nothrow) ResidualWorkspace);
    if (!storage) return cudaErrorMemoryAllocation;
    auto &res = *storage;
    int n = work.ice + work.panels;
    int nnz = work.head_nnz + work.tail_nnz;
    CUDA(cudaMalloc(&res.diagonal_positions, sizeof(int) * size_t(work.ice)));
    CUDA(cudaMalloc(&res.diagonal_values, sizeof(Complex) * size_t(work.ice)));
    CUDA(cudaMalloc(&res.diagonal_product, sizeof(Complex) * size_t(work.ice)));
    CUDA(cudaMalloc(&res.applied, sizeof(Complex) * size_t(n)));
    CUDA(cudaMemcpyAsync(res.diagonal_positions, positions, sizeof(int) * size_t(work.ice),
                         cudaMemcpyHostToDevice, work.stream));
    BLAS(cublasCreate(&res.blas));
    BLAS(cublasSetStream(res.blas, work.stream));
    SPARSE(cusparseCreate(&res.sparse));
    SPARSE(cusparseSetStream(res.sparse, work.stream));
    SPARSE(cusparseCreateCsr(&res.upper, n, n, nnz, work.row, work.column, work.values,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
    SPARSE(cusparseCreateDnVec(&res.x, n, work.solution_values, CUDA_C_64F));
    SPARSE(cusparseCreateDnVec(&res.ax, n, res.applied, CUDA_C_64F));
    SPARSE(cusparseCreateDnVec(&res.values, nnz, work.values, CUDA_C_64F));
    SPARSE(cusparseCreateSpVec(&res.diagonal, nnz, work.ice, res.diagonal_positions, res.diagonal_values,
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
    size_t direct_bytes = 0, transpose_bytes = 0;
    const auto one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0);
    SPARSE(cusparseSpMV_bufferSize(res.sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, res.upper,
                                  res.x, &zero, res.ax, CUDA_C_64F, sparse_algorithm, &direct_bytes));
    SPARSE(cusparseSpMV_bufferSize(res.sparse, CUSPARSE_OPERATION_TRANSPOSE, &one, res.upper,
                                  res.x, &one, res.ax, CUDA_C_64F, sparse_algorithm, &transpose_bytes));
    const size_t bytes = direct_bytes > transpose_bytes ? direct_bytes : transpose_bytes;
    if (bytes) CUDA(cudaMalloc(&res.scratch, bytes));
    CUDA(cudaStreamSynchronize(work.stream));
    work.residual = std::move(storage);
    return 0;
}

int prepare_panel_load(Workspace &work, ResidualWorkspace &res) {
    const int size = work.ice + work.panels;
    CUDA(cudaMalloc(&res.panel_rhs, sizeof(Complex) * size_t(size)));
    SPARSE(cusparseCreateCsr(&res.coupling, work.panels, size, work.head_nnz, work.row, work.column, work.values,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));
    SPARSE(cusparseCreateDnVec(&res.panel_values, work.panels, work.rhs_values, CUDA_C_64F));
    SPARSE(cusparseCreateDnVec(&res.panel_rhs_values, size, res.panel_rhs, CUDA_C_64F));
    size_t direct_bytes = 0, transpose_bytes = 0;
    const auto one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0);
    SPARSE(cusparseSpMV_bufferSize(res.sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, res.coupling,
                                  res.x, &zero, res.panel_values, CUDA_C_64F, sparse_algorithm, &direct_bytes));
    SPARSE(cusparseSpMV_bufferSize(res.sparse, CUSPARSE_OPERATION_TRANSPOSE, &one, res.coupling,
                                  res.panel_values, &zero, res.panel_rhs_values, CUDA_C_64F,
                                  sparse_algorithm, &transpose_bytes));
    const size_t bytes = direct_bytes > transpose_bytes ? direct_bytes : transpose_bytes;
    if (bytes) CUDA(cudaMalloc(&res.panel_scratch, bytes));
    return 0;
}

int recover_ice(Workspace &w, cudssMatrix_t rhs, cudssMatrix_t solution, Complex *rhs_values, int count) {
    DSS(cudssExecute(w.solver, CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD,
                     w.config, w.data, w.matrix, solution, rhs));
    // The partial diagonal solve swaps the roles of the two cuDSS dense buffers.
    DSS(cudssExecute(w.solver, CUDSS_PHASE_SOLVE_DIAG, w.config, w.data, w.matrix, rhs, solution));
    // Prescribe the reordered augmented unknowns to zero.
    const size_t panel_bytes = sizeof(Complex) * size_t(w.panels);
    if (count == 1) {
        CUDA(cudaMemsetAsync(rhs_values + w.ice, 0, panel_bytes, w.stream));
    } else {
        const size_t stride = sizeof(Complex) * (size_t(w.ice) + w.panels);
        CUDA(cudaMemset2DAsync(rhs_values + w.ice, stride, 0, panel_bytes, count, w.stream));
    }
    DSS(cudssExecute(w.solver, CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM,
                     w.config, w.data, w.matrix, solution, rhs));
    return 0;
}

int recover_ice(Workspace &w) {
    return recover_ice(w, w.rhs, w.solution, w.rhs_values, 1);
}

constexpr auto batch_sparse_algorithm = CUSPARSE_SPMM_CSR_ALG1;

int prepare_batch(Workspace &w, int count) {
    if (w.batch && w.batch->count == count) return 0;
    w.batch.reset();
    std::unique_ptr<BatchWorkspace> storage(new (std::nothrow) BatchWorkspace);
    if (!storage) return cudaErrorMemoryAllocation;
    auto &b = *storage;
    auto &r = *w.residual;
    b.count = count;
    const int n = w.ice + w.panels;
    const size_t ice_bytes = sizeof(Complex) * size_t(w.ice) * count;
    const size_t panel_bytes = sizeof(Complex) * size_t(w.panels) * count;
    const size_t system_bytes = ice_bytes + panel_bytes;
    CUDA(cudaMalloc(&b.rhs_values, system_bytes));
    CUDA(cudaMalloc(&b.solution_values, system_bytes));
    CUDA(cudaMalloc(&b.original_rhs, ice_bytes));
    CUDA(cudaMalloc(&b.applied_values, system_bytes));
    CUDA(cudaMalloc(&b.diagonal_product, ice_bytes));
    CUDA(cudaMalloc(&b.normal_values, panel_bytes));
    CUDA(cudaMalloc(&b.norms, 2 * sizeof(double) * size_t(count)));
    try {
        b.host_norms.resize(2 * size_t(count));
    } catch (const std::bad_alloc &) {
        return cudaErrorMemoryAllocation;
    }
    DSS(cudssMatrixCreateDn(&b.rhs, n, count, n, b.rhs_values, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS(cudssMatrixCreateDn(&b.solution, n, count, n, b.solution_values, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    SPARSE(cusparseCreateDnMat(&b.values, n, count, n, b.solution_values, CUDA_C_64F, CUSPARSE_ORDER_COL));
    SPARSE(cusparseCreateDnMat(&b.applied, n, count, n, b.applied_values, CUDA_C_64F, CUSPARSE_ORDER_COL));
    SPARSE(cusparseCreateDnMat(&b.normal, w.panels, count, w.panels, b.normal_values, CUDA_C_64F, CUSPARSE_ORDER_COL));
    BLAS(cublasCreate(&b.norm_blas));
    BLAS(cublasSetStream(b.norm_blas, w.stream));
    BLAS(cublasSetPointerMode(b.norm_blas, CUBLAS_POINTER_MODE_DEVICE));

    const auto direct = CUSPARSE_OPERATION_NON_TRANSPOSE, transposed = CUSPARSE_OPERATION_TRANSPOSE;
    const auto one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0);
    size_t upper_bytes = 0, lower_bytes = 0, normal_bytes = 0;
    SPARSE(cusparseSpMM_bufferSize(r.sparse, direct, direct, &one, r.upper, b.values, &zero, b.applied,
                                   CUDA_C_64F, batch_sparse_algorithm, &upper_bytes));
    SPARSE(cusparseSpMM_bufferSize(r.sparse, transposed, direct, &one, r.upper, b.values, &one, b.applied,
                                   CUDA_C_64F, batch_sparse_algorithm, &lower_bytes));
    SPARSE(cusparseSpMM_bufferSize(r.sparse, direct, direct, &one, r.coupling, b.values, &zero, b.normal,
                                   CUDA_C_64F, batch_sparse_algorithm, &normal_bytes));
    const size_t scratch = std::max({upper_bytes, lower_bytes, normal_bytes});
    if (scratch) CUDA(cudaMalloc(&b.scratch, scratch));
    w.batch = std::move(storage);
    return 0;
}

}  // namespace

extern "C" int gpu_panel_create_configured(int device, int ice, int panels, int head_nnz, int tail_nnz,
                                           const int *row, const int *column, const int *marks, int amd,
                                           int nd_levels, int superpanels, int64_t *stats, void **result) {
    *result = nullptr;
    if (ice < 1 || panels < 1 || head_nnz < 1 || tail_nnz < 0) return cudaErrorInvalidValue;
    CUDA(cudaSetDevice(device));
    std::unique_ptr<Workspace> storage(new (std::nothrow) Workspace);
    if (!storage) return cudaErrorMemoryAllocation;
    auto &w = *storage;
    w.device = device;
    w.ice = ice;
    w.panels = panels;
    w.head_nnz = head_nnz;
    w.tail_nnz = tail_nnz;
    const int n = ice + panels;
    const int nnz = head_nnz + tail_nnz;
    const size_t vector_bytes = sizeof(Complex) * size_t(n);
    CUDA(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    CUDA(cudaMalloc(&w.row, sizeof(int) * (size_t(n) + 1)));
    CUDA(cudaMalloc(&w.column, sizeof(int) * size_t(nnz)));
    CUDA(cudaMalloc(&w.values, sizeof(Complex) * size_t(nnz)));
    CUDA(cudaMalloc(&w.solution_values, vector_bytes));
    CUDA(cudaMalloc(&w.rhs_values, vector_bytes));
    CUDA(cudaMalloc(&w.schur_values, sizeof(Complex) * size_t(panels) * panels));
    for (auto &event : w.events) CUDA(cudaEventCreate(&event));
    CUDA(cudaMemcpyAsync(w.row, row, sizeof(int) * (size_t(n) + 1), cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemcpyAsync(w.column, column, sizeof(int) * size_t(nnz), cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemsetAsync(w.values, 0, sizeof(Complex) * size_t(nnz), w.stream));
    CUDA(cudaMemsetAsync(w.solution_values, 0, vector_bytes, w.stream));
    CUDA(cudaMemsetAsync(w.rhs_values, 0, vector_bytes, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));

    DSS(cudssCreate(&w.solver));
    DSS(cudssSetStream(w.solver, w.stream));
    DSS(cudssConfigCreate(&w.config));
    DSS(cudssDataCreate(w.solver, &w.data));
    int enabled = 1;
    DSS(cudssConfigSet(w.config, CUDSS_CONFIG_SCHUR_MODE, &enabled, sizeof(enabled)));
    if (amd) {
        auto alg = CUDSS_ALG_3;
        DSS(cudssConfigSet(w.config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)));
    }
    if (nd_levels) {
        DSS(cudssConfigSet(w.config, CUDSS_CONFIG_ND_NLEVELS, &nd_levels, sizeof(nd_levels)));
    }
    if (!superpanels) {
        DSS(cudssConfigSet(w.config, CUDSS_CONFIG_USE_SUPERPANELS, &superpanels, sizeof(superpanels)));
    }
    DSS(cudssDataSet(w.solver, w.data, CUDSS_DATA_USER_SCHUR_INDICES, const_cast<int *>(marks), sizeof(int) * size_t(n)));
    DSS(cudssMatrixCreateCsr(&w.matrix, n, n, nnz, w.row, nullptr, w.column, w.values,
                             CUDA_R_32I, CUDA_C_64F, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));
    DSS(cudssMatrixCreateDn(&w.solution, n, 1, n, w.solution_values, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS(cudssMatrixCreateDn(&w.rhs, n, 1, n, w.rhs_values, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS(cudssMatrixCreateDn(&w.schur, panels, panels, panels, w.schur_values, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS(cudssExecute(w.solver, CUDSS_PHASE_ANALYSIS, w.config, w.data, w.matrix, w.solution, w.rhs));
    CUDA(cudaStreamSynchronize(w.stream));
    int64_t shape[3]{};
    size_t used = 0;
    DSS(cudssDataGet(w.solver, w.data, CUDSS_DATA_SCHUR_SHAPE, shape, sizeof(shape), &used));
    if (shape[0] != panels || shape[1] != panels) return 2002;
    if (stats) {
        int64_t estimates[16]{};
        if (cudssDataGet(w.solver, w.data, CUDSS_DATA_MEMORY_ESTIMATES, estimates, sizeof(estimates), &used)
                == CUDSS_STATUS_SUCCESS) stats[0] = estimates[1];
        if (cudssDataGet(w.solver, w.data, CUDSS_DATA_LU_NNZ, stats + 1, sizeof(int64_t), &used)
                != CUDSS_STATUS_SUCCESS) stats[1] = -1;
    }
    *result = storage.release();
    return 0;
}

extern "C" int gpu_panel_create(int device, int ice, int panels, int head_nnz, int tail_nnz,
                                const int *row, const int *column, const int *marks, void **result) {
    return gpu_panel_create_configured(device, ice, panels, head_nnz, tail_nnz, row, column, marks,
                                       0, 0, 1, nullptr, result);
}

extern "C" void gpu_panel_destroy(void *workspace) {
    delete static_cast<Workspace *>(workspace);
}

extern "C" int gpu_panel_factor(void *workspace, const Complex *head, const Complex *tail,
                                Complex *schur, double *times) {
    auto &w = *static_cast<Workspace *>(workspace);
    w.factored = false;
    w.solved = false;
    CUDA(cudaSetDevice(w.device));
    if (times) CUDA(cudaEventRecord(w.events[0], w.stream));
    CUDA(cudaMemcpyAsync(w.values, head, sizeof(Complex) * size_t(w.head_nnz), cudaMemcpyHostToDevice, w.stream));
    if (w.tail_nnz) {
        CUDA(cudaMemcpyAsync(w.values + w.head_nnz, tail, sizeof(Complex) * size_t(w.tail_nnz),
                             cudaMemcpyHostToDevice, w.stream));
    }
    return factor_and_extract(w, schur, times);
}

#ifdef PANEL_DEVICE_ASSEMBLY
extern "C" int gpu_panel_assembly_setup(void *workspace, const double *stiffness, const double *mass,
                                        int groups, const int *targets, const int *offsets, int contributions,
                                        const int *sources, const int *rows, int degree, int coefficients, int levels) {
    auto &w = *static_cast<Workspace *>(workspace);
    CUDA(cudaSetDevice(w.device));
    std::unique_ptr<AssemblyWorkspace> storage(new (std::nothrow) AssemblyWorkspace);
    if (!storage) return cudaErrorMemoryAllocation;
    auto &a = *storage;
    a.groups = groups;
    a.degree = degree;
    a.coefficients = coefficients;
    a.levels = levels;
    if (const int code = copy_static(a.stiffness, stiffness, w.tail_nnz, w.stream)) return code;
    if (const int code = copy_static(a.mass, mass, w.tail_nnz, w.stream)) return code;
    if (const int code = copy_static(a.targets, targets, groups, w.stream)) return code;
    if (const int code = copy_static(a.offsets, offsets, size_t(groups) + 1, w.stream)) return code;
    if (const int code = copy_static(a.sources, sources, contributions, w.stream)) return code;
    if (const int code = copy_static(a.rows, rows, contributions, w.stream)) return code;
    if (coefficients) CUDA(cudaMalloc(&a.pml, sizeof(Complex) * size_t(coefficients) * levels));
    CUDA(cudaStreamSynchronize(w.stream));
    w.assembly = std::move(storage);
    return 0;
}

extern "C" int gpu_panel_factor_assembled(void *workspace, const Complex *head, double wave_re, double wave_im,
                                           const Complex *constant, const Complex *linear, const Complex *quadratic,
                                           Complex *schur, double *times) {
    auto &w = *static_cast<Workspace *>(workspace);
    if (!w.assembly) return cudaErrorInvalidValue;
    auto &a = *w.assembly;
    w.factored = false;
    w.solved = false;
    CUDA(cudaSetDevice(w.device));
    if (times) CUDA(cudaEventRecord(w.events[0], w.stream));
    CUDA(cudaMemcpyAsync(w.values, head, sizeof(Complex) * size_t(w.head_nnz), cudaMemcpyHostToDevice, w.stream));
    const size_t bytes = sizeof(Complex) * size_t(a.coefficients);
    if (bytes) {
        CUDA(cudaMemcpyAsync(a.pml, constant, bytes, cudaMemcpyHostToDevice, w.stream));
        if (a.levels == 3) {
            CUDA(cudaMemcpyAsync(a.pml + a.coefficients, linear, bytes, cudaMemcpyHostToDevice, w.stream));
            CUDA(cudaMemcpyAsync(a.pml + 2 * size_t(a.coefficients), quadratic, bytes, cudaMemcpyHostToDevice, w.stream));
        }
    }
    CUDA(launch_panel_assembly(w.stream, w.tail_nnz, a.stiffness, a.mass, wave_re, wave_im,
                               a.groups, a.targets, a.offsets, a.sources, a.rows, a.degree,
                               a.coefficients, a.pml, w.values + w.head_nnz));
    return factor_and_extract(w, schur, times);
}

extern "C" int gpu_panel_read_elastic_values(void *workspace, Complex *values) {
    auto &w = *static_cast<Workspace *>(workspace);
    if (!w.factored) return cudaErrorInvalidValue;
    CUDA(cudaSetDevice(w.device));
    CUDA(cudaMemcpyAsync(values, w.values + w.head_nnz, sizeof(Complex) * size_t(w.tail_nnz),
                         cudaMemcpyDeviceToHost, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    return 0;
}
#endif

extern "C" int gpu_panel_solve(void *workspace, const Complex *rhs, Complex *solution) {
    auto &w = *static_cast<Workspace *>(workspace);
    if (!w.factored) return cudaErrorInvalidValue;
    w.solved = false;
    w.panel_solved = false;
    CUDA(cudaSetDevice(w.device));
    const size_t ice_bytes = sizeof(Complex) * size_t(w.ice);
    const size_t panel_bytes = sizeof(Complex) * size_t(w.panels);
    CUDA(cudaMemcpyAsync(w.rhs_values + w.panels, rhs, ice_bytes, cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemsetAsync(w.rhs_values, 0, panel_bytes, w.stream));
    if (const int status = recover_ice(w)) return status;
    CUDA(cudaMemcpyAsync(solution, w.solution_values + w.panels, ice_bytes, cudaMemcpyDeviceToHost, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    w.solved = true;
    return 0;
}

extern "C" int gpu_panel_solve_normal(void *workspace, const Complex *pressure, const int *diagonal_positions,
                                      Complex *unscaled_normal) {
    auto &w = *static_cast<Workspace *>(workspace);
    if (!w.factored) return cudaErrorInvalidValue;
    w.solved = false;
    w.panel_solved = false;
    CUDA(cudaSetDevice(w.device));
    if (!w.residual) {
        if (const int status = create_residual(w, diagonal_positions)) return status;
    }
    auto &res = *w.residual;
    if (!res.panel_rhs) {
        if (const int status = prepare_panel_load(w, res)) return status;
    }
    const size_t panel_bytes = sizeof(Complex) * size_t(w.panels);
    const size_t vector_bytes = sizeof(Complex) * (size_t(w.ice) + w.panels);
    const auto one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0);
    CUDA(cudaMemcpyAsync(w.rhs_values, pressure, panel_bytes, cudaMemcpyHostToDevice, w.stream));
    SPARSE(cusparseSpMV(res.sparse, CUSPARSE_OPERATION_TRANSPOSE, &one, res.coupling, res.panel_values,
                         &zero, res.panel_rhs_values, CUDA_C_64F, sparse_algorithm, res.panel_scratch));
    CUDA(cudaMemcpyAsync(w.rhs_values, res.panel_rhs, vector_bytes, cudaMemcpyDeviceToDevice, w.stream));
    if (const int status = recover_ice(w)) return status;
    SPARSE(cusparseSpMV(res.sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, res.coupling, res.x,
                         &zero, res.panel_values, CUDA_C_64F, sparse_algorithm, res.panel_scratch));
    CUDA(cudaMemcpyAsync(unscaled_normal, w.rhs_values, panel_bytes, cudaMemcpyDeviceToHost, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    w.solved = w.panel_solved = true;
    return 0;
}

static int solve_normal_many(void *workspace, int count, const Complex *rhs, cudaMemcpyKind input_copy,
                             const int *diagonal_positions, Complex *normal, double *residuals) {
    auto &w = *static_cast<Workspace *>(workspace);
    if (!w.factored || count < 1 || !rhs || !diagonal_positions || !normal || !residuals)
        return cudaErrorInvalidValue;
    w.solved = w.panel_solved = false;
    CUDA(cudaSetDevice(w.device));
    if (!w.residual) {
        if (const int status = create_residual(w, diagonal_positions)) return status;
    }
    auto &r = *w.residual;
    if (!r.panel_rhs) {
        if (const int status = prepare_panel_load(w, r)) return status;
    }
    if (const int status = prepare_batch(w, count)) return status;
    auto &b = *w.batch;
    const int n = w.ice + w.panels;
    const size_t stride = sizeof(Complex) * size_t(n);
    const size_t panel_bytes = sizeof(Complex) * size_t(w.panels);
    const size_t ice_bytes = sizeof(Complex) * size_t(w.ice);
    const Complex *original_rhs = rhs;
    if (input_copy == cudaMemcpyHostToDevice) {
        CUDA(cudaMemcpyAsync(b.original_rhs, rhs, ice_bytes * count, input_copy, w.stream));
        original_rhs = b.original_rhs;
    }
    CUDA(cudaMemset2DAsync(b.rhs_values, stride, 0, panel_bytes, count, w.stream));
    CUDA(cudaMemcpy2DAsync(b.rhs_values + w.panels, stride, original_rhs, ice_bytes, ice_bytes, count,
                           cudaMemcpyDeviceToDevice, w.stream));
    if (const int status = recover_ice(w, b.rhs, b.solution, b.rhs_values, count)) return status;
    CUDA(cudaMemset2DAsync(b.solution_values, stride, 0, panel_bytes, count, w.stream));

    // Recover L.T @ u and check A @ u against every original elastic RHS.
    const auto direct = CUSPARSE_OPERATION_NON_TRANSPOSE, transposed = CUSPARSE_OPERATION_TRANSPOSE;
    const auto one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0), neg = make_cuDoubleComplex(-1, 0);
    SPARSE(cusparseSpMM(r.sparse, direct, direct, &one, r.coupling, b.values, &zero, b.normal,
                        CUDA_C_64F, batch_sparse_algorithm, b.scratch));
    SPARSE(cusparseSpMM(r.sparse, direct, direct, &one, r.upper, b.values, &zero, b.applied,
                        CUDA_C_64F, batch_sparse_algorithm, b.scratch));
    SPARSE(cusparseSpMM(r.sparse, transposed, direct, &one, r.upper, b.values, &one, b.applied,
                        CUDA_C_64F, batch_sparse_algorithm, b.scratch));
    SPARSE(cusparseGather(r.sparse, r.values, r.diagonal));
    BLAS(cublasZdgmm(r.blas, CUBLAS_SIDE_LEFT, w.ice, count, b.solution_values + w.panels, n,
                     r.diagonal_values, 1, b.diagonal_product, w.ice));
    for (int column = 0; column < count; ++column) {
        auto *residual = b.applied_values + size_t(n) * column + w.panels;
        const auto *diagonal = b.diagonal_product + size_t(w.ice) * column;
        const auto *original = original_rhs + size_t(w.ice) * column;
        BLAS(cublasZaxpy(r.blas, w.ice, &neg, diagonal, 1, residual, 1));
        BLAS(cublasZaxpy(r.blas, w.ice, &neg, original, 1, residual, 1));
        BLAS(cublasDznrm2(b.norm_blas, w.ice, residual, 1, b.norms + column));
        BLAS(cublasDznrm2(b.norm_blas, w.ice, original, 1, b.norms + count + column));
    }
    CUDA(cudaMemcpyAsync(normal, b.normal_values, panel_bytes * count, cudaMemcpyDeviceToHost, w.stream));
    CUDA(cudaMemcpyAsync(b.host_norms.data(), b.norms, 2 * sizeof(double) * size_t(count),
                         cudaMemcpyDeviceToHost, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    for (int column = 0; column < count; ++column) {
        const double error = b.host_norms[column], norm = b.host_norms[count + column];
        residuals[column] = norm ? error / norm : error;
    }
    return 0;
}

extern "C" int gpu_panel_solve_normal_many(void *workspace, int count, const Complex *rhs,
                                          const int *diagonal_positions, Complex *normal, double *residuals) {
    return solve_normal_many(workspace, count, rhs, cudaMemcpyHostToDevice, diagonal_positions, normal, residuals);
}

extern "C" int gpu_panel_solve_normal_many_device(void *workspace, int count, const Complex *rhs,
                                                 const int *diagonal_positions, Complex *normal, double *residuals) {
    return solve_normal_many(workspace, count, rhs, cudaMemcpyDeviceToDevice, diagonal_positions, normal, residuals);
}

extern "C" int gpu_panel_original_residual(void *workspace, const Complex *rhs, const int *diagonal_positions,
                                          double *result) {
    auto &w = *static_cast<Workspace *>(workspace);
    if (!w.solved) return cudaErrorInvalidValue;
    CUDA(cudaSetDevice(w.device));
    if (!w.residual) {
        if (const int status = create_residual(w, diagonal_positions)) return status;
    }
    auto &res = *w.residual;
    const size_t ice_bytes = sizeof(Complex) * size_t(w.ice);
    if (!rhs && !w.panel_solved) return cudaErrorInvalidValue;
    const Complex *original_rhs = rhs ? w.rhs_values + w.panels : res.panel_rhs + w.panels;
    if (rhs) CUDA(cudaMemcpyAsync(w.rhs_values + w.panels, rhs, ice_bytes, cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemsetAsync(w.solution_values, 0, sizeof(Complex) * size_t(w.panels), w.stream));
    const auto one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0), neg = make_cuDoubleComplex(-1, 0);
    SPARSE(cusparseSpMV(res.sparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, res.upper,
                         res.x, &zero, res.ax, CUDA_C_64F, sparse_algorithm, res.scratch));
    SPARSE(cusparseSpMV(res.sparse, CUSPARSE_OPERATION_TRANSPOSE, &one, res.upper,
                         res.x, &one, res.ax, CUDA_C_64F, sparse_algorithm, res.scratch));
    SPARSE(cusparseGather(res.sparse, res.values, res.diagonal));
    BLAS(cublasZdgmm(res.blas, CUBLAS_SIDE_LEFT, w.ice, 1, w.solution_values + w.panels, w.ice,
                     res.diagonal_values, 1, res.diagonal_product, w.ice));
    BLAS(cublasZaxpy(res.blas, w.ice, &neg, res.diagonal_product, 1, res.applied + w.panels, 1));
    BLAS(cublasZaxpy(res.blas, w.ice, &neg, original_rhs, 1, res.applied + w.panels, 1));
    double error = 0, norm = 0;
    BLAS(cublasDznrm2(res.blas, w.ice, res.applied + w.panels, 1, &error));
    BLAS(cublasDznrm2(res.blas, w.ice, original_rhs, 1, &norm));
    *result = norm ? error / norm : error;
    return 0;
}
