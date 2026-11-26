#include <cstdio>
#include <memory>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cudss.h>
#include <cusolverDn.h>

namespace {

struct Workspace {
    int device, n, trace_count, panels;
    cudaStream_t stream = nullptr;
    cudssHandle_t sparse = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cusolverDnHandle_t dense = nullptr;
    cudssMatrix_t a = nullptr, x = nullptr, b = nullptr, schur = nullptr;
    int *row = nullptr, *column = nullptr, *pivots = nullptr, *info = nullptr;
    cuDoubleComplex *values = nullptr, *x_values = nullptr, *b_values = nullptr;
    cuDoubleComplex *schur_values = nullptr, *trace_values = nullptr, *buffer = nullptr;
    bool factored = false;

    Workspace(int device_, int n_, int trace_count_, int panels_)
        : device(device_), n(n_), trace_count(trace_count_), panels(panels_) {}

    ~Workspace() {
        if (cudaSetDevice(device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);
        if (schur) cudssMatrixDestroy(schur);
        if (b) cudssMatrixDestroy(b);
        if (x) cudssMatrixDestroy(x);
        if (a) cudssMatrixDestroy(a);
        if (data) cudssDataDestroy(sparse, data);
        if (config) cudssConfigDestroy(config);
        if (sparse) cudssDestroy(sparse);
        if (dense) cusolverDnDestroy(dense);
        if (buffer) cudaFree(buffer);
        if (trace_values) cudaFree(trace_values);
        if (schur_values) cudaFree(schur_values);
        if (b_values) cudaFree(b_values);
        if (x_values) cudaFree(x_values);
        if (info) cudaFree(info);
        if (pivots) cudaFree(pivots);
        if (values) cudaFree(values);
        if (column) cudaFree(column);
        if (row) cudaFree(row);
        if (stream) cudaStreamDestroy(stream);
    }
};

#define CUDA(call) do { const auto status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); return int(status); } } while (0)
#define DSS(call) do { const auto status = (call); if (status != CUDSS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s: cuDSS status %d\n", #call, int(status)); return 1000 + int(status); } } while (0)
#define DENSE(call) do { const auto status = (call); if (status != CUSOLVER_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s: cuSOLVER status %d\n", #call, int(status)); return 2000 + int(status); } } while (0)

int check_dense_info(Workspace &work) {
    int info = 0;
    CUDA(cudaMemcpyAsync(&info, work.info, sizeof(info), cudaMemcpyDeviceToHost, work.stream));
    CUDA(cudaStreamSynchronize(work.stream));
    if (info) {
        std::fprintf(stderr, "cuSOLVER numeric status %d\n", info);
        return 3001;
    }
    return 0;
}

int check_sparse_info(Workspace &work) {
    size_t used = 0;
    int info = 0;
    DSS(cudssDataGet(work.sparse, work.data, CUDSS_DATA_INFO, &info, sizeof(info), &used));
    if (info) {
        std::fprintf(stderr, "cuDSS numeric status %d\n", info);
        return 3002;
    }
    return 0;
}

} // namespace

extern "C" int gpu_trace_create(int device, int n, int nnz, int trace_count, int panels,
                                const int *row, const int *column, const cuDoubleComplex *values,
                                const int *marks, void **result) {
    *result = nullptr;
    CUDA(cudaSetDevice(device));
    auto workspace = std::make_unique<Workspace>(device, n, trace_count, panels);
    auto &w = *workspace;
    const auto complex_size = sizeof(cuDoubleComplex);
    CUDA(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    CUDA(cudaMalloc(&w.row, sizeof(int) * (n + 1)));
    CUDA(cudaMalloc(&w.column, sizeof(int) * nnz));
    CUDA(cudaMalloc(&w.values, complex_size * nnz));
    CUDA(cudaMalloc(&w.x_values, complex_size * n));
    CUDA(cudaMalloc(&w.b_values, complex_size * n));
    CUDA(cudaMalloc(&w.schur_values, complex_size * size_t(trace_count) * trace_count));
    CUDA(cudaMalloc(&w.trace_values, complex_size * size_t(trace_count) * panels));
    CUDA(cudaMalloc(&w.pivots, sizeof(int) * trace_count));
    CUDA(cudaMalloc(&w.info, sizeof(int)));
    CUDA(cudaMemcpyAsync(w.row, row, sizeof(int) * (n + 1), cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemcpyAsync(w.column, column, sizeof(int) * nnz, cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemcpyAsync(w.values, values, complex_size * nnz, cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemsetAsync(w.x_values, 0, complex_size * n, w.stream));
    CUDA(cudaMemsetAsync(w.b_values, 0, complex_size * n, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    DSS(cudssCreate(&w.sparse));
    DSS(cudssSetStream(w.sparse, w.stream));
    DSS(cudssConfigCreate(&w.config));
    DSS(cudssDataCreate(w.sparse, &w.data));
    int enabled = 1;
    DSS(cudssConfigSet(w.config, CUDSS_CONFIG_SCHUR_MODE, &enabled, sizeof(enabled)));
    DSS(cudssDataSet(w.sparse, w.data, CUDSS_DATA_USER_SCHUR_INDICES, const_cast<int *>(marks), sizeof(int) * n));
    DSS(cudssMatrixCreateCsr(&w.a, n, n, nnz, w.row, nullptr, w.column, w.values, CUDA_R_32I, CUDA_C_64F,
                             CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));
    DSS(cudssMatrixCreateDn(&w.x, n, 1, n, w.x_values, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS(cudssMatrixCreateDn(&w.b, n, 1, n, w.b_values, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS(cudssMatrixCreateDn(&w.schur, trace_count, trace_count, trace_count, w.schur_values,
                            CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS(cudssExecute(w.sparse, CUDSS_PHASE_ANALYSIS, w.config, w.data, w.a, w.x, w.b));
    CUDA(cudaStreamSynchronize(w.stream));
    int64_t shape[3]{};
    size_t used = 0;
    DSS(cudssDataGet(w.sparse, w.data, CUDSS_DATA_SCHUR_SHAPE, shape, sizeof(shape), &used));
    if (shape[0] != trace_count || shape[1] != trace_count) return 3003;
    DENSE(cusolverDnCreate(&w.dense));
    DENSE(cusolverDnSetStream(w.dense, w.stream));
    int buffer_size = 0;
    DENSE(cusolverDnZgetrf_bufferSize(w.dense, trace_count, trace_count, w.schur_values, trace_count, &buffer_size));
    CUDA(cudaMalloc(&w.buffer, complex_size * buffer_size));
    *result = workspace.release();
    return 0;
}

extern "C" void gpu_trace_destroy(void *workspace) {
    delete static_cast<Workspace *>(workspace);
}

extern "C" int gpu_trace_factor(void *workspace, int nnz, const cuDoubleComplex *values,
                                const cuDoubleComplex *trace_load, cuDoubleComplex *response) {
    auto &w = *static_cast<Workspace *>(workspace);
    w.factored = false;
    CUDA(cudaSetDevice(w.device));
    const int m = w.trace_count;
    const auto trace_size = sizeof(cuDoubleComplex) * size_t(m) * w.panels;
    CUDA(cudaMemcpyAsync(w.values, values, sizeof(cuDoubleComplex) * nnz, cudaMemcpyHostToDevice, w.stream));
    CUDA(cudaMemcpyAsync(w.trace_values, trace_load, trace_size, cudaMemcpyHostToDevice, w.stream));
    DSS(cudssExecute(w.sparse, CUDSS_PHASE_FACTORIZATION, w.config, w.data, w.a, w.x, w.b));
    CUDA(cudaStreamSynchronize(w.stream));
    const int sparse_info = check_sparse_info(w);
    if (sparse_info) return sparse_info;
    size_t used = 0;
    DSS(cudssDataGet(w.sparse, w.data, CUDSS_DATA_SCHUR_MATRIX, &w.schur, sizeof(w.schur), &used));
    DENSE(cusolverDnZgetrf(w.dense, m, m, w.schur_values, m, w.buffer, w.pivots, w.info));
    const int factor_info = check_dense_info(w);
    if (factor_info) return factor_info;
    DENSE(cusolverDnZgetrs(w.dense, CUBLAS_OP_N, m, w.panels, w.schur_values, m, w.pivots, w.trace_values, m, w.info));
    const int solve_info = check_dense_info(w);
    if (solve_info) return solve_info;
    CUDA(cudaMemcpyAsync(response, w.trace_values, trace_size, cudaMemcpyDeviceToHost, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    w.factored = true;
    return 0;
}

extern "C" int gpu_trace_solve(void *workspace, const cuDoubleComplex *rhs, cuDoubleComplex *solution) {
    auto &w = *static_cast<Workspace *>(workspace);
    if (!w.factored) return 3004;
    CUDA(cudaSetDevice(w.device));
    const auto vector_size = sizeof(cuDoubleComplex) * w.n;
    CUDA(cudaMemcpyAsync(w.b_values, rhs, vector_size, cudaMemcpyHostToDevice, w.stream));
    DSS(cudssExecute(w.sparse, CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD,
                      w.config, w.data, w.a, w.x, w.b));
    DSS(cudssExecute(w.sparse, CUDSS_PHASE_SOLVE_DIAG, w.config, w.data, w.a, w.b, w.x));
    DENSE(cusolverDnZgetrs(w.dense, CUBLAS_OP_N, w.trace_count, 1, w.schur_values, w.trace_count,
                          w.pivots, w.b_values + (w.n - w.trace_count), w.n, w.info));
    const int dense_info = check_dense_info(w);
    if (dense_info) return dense_info;
    DSS(cudssExecute(w.sparse, CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM,
                      w.config, w.data, w.a, w.x, w.b));
    CUDA(cudaMemcpyAsync(solution, w.x_values, vector_size, cudaMemcpyDeviceToHost, w.stream));
    CUDA(cudaStreamSynchronize(w.stream));
    return 0;
}
