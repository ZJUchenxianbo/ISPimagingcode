#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cudss.h>

namespace {

struct Workspace {
    explicit Workspace(int device_id) : device(device_id) {}
    int device;
    cudaStream_t stream = nullptr;
    cudssHandle_t solver = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matrix = nullptr, rhs = nullptr, solution = nullptr;
    void *offsets = nullptr, *columns = nullptr;
    cuDoubleComplex *values = nullptr, *b = nullptr, *x = nullptr;
    int64_t n = 0, nnz = 0;

    ~Workspace() {
        if (cudaSetDevice(device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);
        if (solution) cudssMatrixDestroy(solution);
        if (rhs) cudssMatrixDestroy(rhs);
        if (matrix) cudssMatrixDestroy(matrix);
        if (data) cudssDataDestroy(solver, data);
        if (config) cudssConfigDestroy(config);
        if (solver) cudssDestroy(solver);
        cudaFree(x);
        cudaFree(b);
        cudaFree(values);
        cudaFree(columns);
        cudaFree(offsets);
        if (stream) cudaStreamDestroy(stream);
    }
};

#define CUDA_TRY(call) do { const auto status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); return int(status); } } while (0)
#define DSS_TRY(call) do { const auto status = (call); if (status != CUDSS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s: cuDSS status %d\n", #call, int(status)); return 1000 + int(status); } } while (0)

template <class Index>
int64_t pack_upper(int64_t n, const Index *offsets, const Index *columns, const cuDoubleComplex *values,
                   int64_t capacity, Index *upper_offsets, Index *upper_columns,
                   cuDoubleComplex *upper_values, bool reuse) {
    int64_t end = 0;
    for (int64_t row = 0; row < n; ++row) {
        const auto *first = std::lower_bound(columns + offsets[row], columns + offsets[row + 1], row);
        const int64_t count = columns + offsets[row + 1] - first;
        if (upper_values) {
            if (count > capacity - end || (reuse && upper_offsets[row] != end)) return -1;
            if (reuse && !std::equal(first, first + count, upper_columns + end)) return -1;
            if (!reuse) {
                upper_offsets[row] = end;
                std::copy_n(first, count, upper_columns + end);
            }
            std::copy_n(values + (first - columns), count, upper_values + end);
        }
        end += count;
    }
    if (upper_values && !reuse) upper_offsets[n] = end;
    return end;
}

int check_numeric(Workspace &work, const char *stage) {
    int info = 0;
    size_t used = 0;
    DSS_TRY(cudssDataGet(work.solver, work.data, CUDSS_DATA_INFO, &info, sizeof(info), &used));
    if (info) {
        std::fprintf(stderr, "%s: cuDSS device indicator %d\n", stage, info);
        return 2000;
    }
    return 0;
}
} // namespace

extern "C" int64_t planar_cudss_upper(int64_t n, int bytes, const void *offsets, const void *columns,
                                      const void *values, int64_t capacity, void *upper_offsets,
                                      void *upper_columns, void *upper_values, int reuse) {
    const auto *in = static_cast<const cuDoubleComplex *>(values);
    auto *out = static_cast<cuDoubleComplex *>(upper_values);
    if (bytes == 4)
        return pack_upper(n, static_cast<const int32_t *>(offsets), static_cast<const int32_t *>(columns),
                          in, capacity, static_cast<int32_t *>(upper_offsets),
                          static_cast<int32_t *>(upper_columns), out, reuse);
    if (bytes == 8)
        return pack_upper(n, static_cast<const int64_t *>(offsets), static_cast<const int64_t *>(columns),
                          in, capacity, static_cast<int64_t *>(upper_offsets),
                          static_cast<int64_t *>(upper_columns), out, reuse);
    return -2;
}

extern "C" int planar_cudss_create(int device, int bytes, int64_t n, int64_t nnz, const void *offsets,
                                   const void *columns, const void *values, int hybrid_memory,
                                   int64_t memory_limit, int refinement_steps, int64_t *memory,
                                   void **workspace) {
    *workspace = nullptr;
    auto owner = std::unique_ptr<Workspace>(new (std::nothrow) Workspace(device));
    if (!owner || (bytes != 4 && bytes != 8)) return 2001;
    auto &work = *owner;
    const size_t complex_size = sizeof(cuDoubleComplex);
    work.n = n;
    work.nnz = nnz;
    CUDA_TRY(cudaSetDevice(device));
    CUDA_TRY(cudaStreamCreateWithFlags(&work.stream, cudaStreamNonBlocking));
    CUDA_TRY(cudaMalloc(&work.offsets, size_t(bytes) * (n + 1)));
    CUDA_TRY(cudaMalloc(&work.columns, size_t(bytes) * nnz));
    CUDA_TRY(cudaMalloc(&work.values, complex_size * nnz));
    CUDA_TRY(cudaMalloc(&work.b, complex_size * n));
    CUDA_TRY(cudaMalloc(&work.x, complex_size * n));
    CUDA_TRY(cudaMemcpyAsync(work.offsets, offsets, size_t(bytes) * (n + 1), cudaMemcpyHostToDevice, work.stream));
    CUDA_TRY(cudaMemcpyAsync(work.columns, columns, size_t(bytes) * nnz, cudaMemcpyHostToDevice, work.stream));
    CUDA_TRY(cudaMemcpyAsync(work.values, values, complex_size * nnz, cudaMemcpyHostToDevice, work.stream));
    CUDA_TRY(cudaMemsetAsync(work.b, 0, complex_size * n, work.stream));
    CUDA_TRY(cudaMemsetAsync(work.x, 0, complex_size * n, work.stream));
    // By default host analysis and device factor/solve. Hybrid memory leaves
    // device factor/solve enabled and only changes the location of factor storage.
    DSS_TRY(cudssCreate(&work.solver));
    DSS_TRY(cudssSetStream(work.solver, work.stream));
    DSS_TRY(cudssConfigCreate(&work.config));
    DSS_TRY(cudssDataCreate(work.solver, &work.data));
    DSS_TRY(cudssConfigSet(work.config, CUDSS_CONFIG_IR_N_STEPS, &refinement_steps, sizeof(refinement_steps)));
    if (hybrid_memory) {
        DSS_TRY(cudssConfigSet(work.config, CUDSS_CONFIG_HYBRID_MODE, &hybrid_memory, sizeof(hybrid_memory)));
        if (memory_limit > 0)
            DSS_TRY(cudssConfigSet(work.config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &memory_limit, sizeof(memory_limit)));
    }
    const auto index_type = bytes == 4 ? CUDA_R_32I : CUDA_R_64I;
    DSS_TRY(cudssMatrixCreateCsr(&work.matrix, n, n, nnz, work.offsets, nullptr, work.columns, work.values,
                                index_type, CUDA_C_64F, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));
    DSS_TRY(cudssMatrixCreateDn(&work.rhs, n, 1, n, work.b, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS_TRY(cudssMatrixCreateDn(&work.solution, n, 1, n, work.x, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDA_TRY(cudaStreamSynchronize(work.stream));
    DSS_TRY(cudssExecute(work.solver, CUDSS_PHASE_ANALYSIS, work.config, work.data, work.matrix, work.solution, work.rhs));
    CUDA_TRY(cudaStreamSynchronize(work.stream));
    if (const int info = check_numeric(work, "analysis")) return info;
    size_t used = 0;
    if (cudssDataGet(work.solver, work.data, CUDSS_DATA_MEMORY_ESTIMATES, memory, 16 * sizeof(int64_t), &used) != CUDSS_STATUS_SUCCESS)
        std::fill_n(memory, 16, int64_t(-1));
    size_t free = 0, total = 0;
    CUDA_TRY(cudaMemGetInfo(&free, &total));
    memory[16] = free;
    memory[17] = total;
    *workspace = owner.release();
    return 0;
}

extern "C" int planar_cudss_factor(void *workspace, const void *values, int64_t *stats) {
    auto &work = *static_cast<Workspace *>(workspace);
    CUDA_TRY(cudaSetDevice(work.device));
    if (values) CUDA_TRY(cudaMemcpyAsync(work.values, values, sizeof(cuDoubleComplex) * work.nnz,
                                        cudaMemcpyHostToDevice, work.stream));
    DSS_TRY(cudssExecute(work.solver, CUDSS_PHASE_FACTORIZATION, work.config, work.data, work.matrix, work.solution, work.rhs));
    CUDA_TRY(cudaStreamSynchronize(work.stream));
    if (const int info = check_numeric(work, "factor")) return info;
    size_t used = 0;
    if (cudssDataGet(work.solver, work.data, CUDSS_DATA_LU_NNZ, stats, sizeof(stats[0]), &used) != CUDSS_STATUS_SUCCESS)
        stats[0] = -1;
    size_t free = 0, total = 0;
    CUDA_TRY(cudaMemGetInfo(&free, &total));
    stats[1] = free;
    stats[2] = total;
    return 0;
}

extern "C" int planar_cudss_solve(void *workspace, const void *rhs, void *solution) {
    auto &work = *static_cast<Workspace *>(workspace);
    CUDA_TRY(cudaSetDevice(work.device));
    const auto bytes = sizeof(cuDoubleComplex) * work.n;
    CUDA_TRY(cudaMemcpyAsync(work.b, rhs, bytes, cudaMemcpyHostToDevice, work.stream));
    DSS_TRY(cudssExecute(work.solver, CUDSS_PHASE_SOLVE, work.config, work.data, work.matrix, work.solution, work.rhs));
    CUDA_TRY(cudaMemcpyAsync(solution, work.x, bytes, cudaMemcpyDeviceToHost, work.stream));
    CUDA_TRY(cudaStreamSynchronize(work.stream));
    return check_numeric(work, "solve");
}

extern "C" void planar_cudss_destroy(void *workspace) {
    delete static_cast<Workspace *>(workspace);
}
