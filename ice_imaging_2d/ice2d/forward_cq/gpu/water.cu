#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>

namespace {

using Complex = cuDoubleComplex;

struct Workspace {
    int device = 0;
    int rows = 0;
    int wet = 0;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    Complex *boundary = nullptr;
    Complex *coupling = nullptr;
    Complex *vector = nullptr;
    Complex *normal = nullptr;
    Complex *result = nullptr;
    Complex *factor = nullptr;
    int *pivots = nullptr;
    int *info = nullptr;
    bool factored = false;

    ~Workspace() {
        cudaSetDevice(device);
        if (blas) cublasDestroy(blas);
        if (solver) cusolverDnDestroy(solver);
        cudaFree(boundary);
        cudaFree(coupling);
        cudaFree(vector);
        cudaFree(normal);
        cudaFree(result);
        cudaFree(factor);
        cudaFree(pivots);
        cudaFree(info);
    }
};

struct WorkBuffer {
    Complex *values = nullptr;
    ~WorkBuffer() { cudaFree(values); }
};

enum class FactorPrecision { none, single, double_precision };

struct BatchWorkspace {
    int capacity = 0;
    Complex *rhs = nullptr;
    Complex *solution = nullptr;
    Complex *defect = nullptr;
    Complex *normal = nullptr;
    cuComplex *correction = nullptr;
    double *relative = nullptr;
    int *state = nullptr;
    int *pending = nullptr;

    ~BatchWorkspace() {
        cudaFree(pending);
        cudaFree(state);
        cudaFree(relative);
        cudaFree(correction);
        cudaFree(normal);
        cudaFree(defect);
        cudaFree(solution);
        cudaFree(rhs);
    }
};

struct DirectWorkspace {
    int device;
    int rows;
    int wet;
    bool gemm3m = false;
    bool fast_refinement = false;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cusolverDnIRSParams_t refinement = nullptr;
    cusolverDnIRSInfos_t refinement_info = nullptr;
    std::size_t refinement_bytes = 0;
    int refinement_iterations = 0;
    bool refinement_accepted = false;
    double refinement_residual = std::numeric_limits<double>::quiet_NaN();
    FactorPrecision factor_precision = FactorPrecision::none;
    const Complex *prepared_boundary = nullptr;
    const Complex *prepared_coupling = nullptr;
    Complex *factor = nullptr;
    Complex *scratch = nullptr;
    Complex *uploaded_coupling = nullptr;
    Complex *response = nullptr;
    Complex *vector = nullptr;
    Complex *normal = nullptr;
    Complex *rhs = nullptr;
    Complex *residual = nullptr;
    int *pivots = nullptr;
    int *info = nullptr;
    double *norms = nullptr;
    std::unique_ptr<BatchWorkspace> batch;
    std::size_t scratch_bytes = 0;

    DirectWorkspace(int device_, int rows_, int wet_) : device(device_), rows(rows_), wet(wet_) {}

    ~DirectWorkspace() {
        if (cudaSetDevice(device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);
        batch.reset();
        if (refinement_info) cusolverDnIRSInfosDestroy(refinement_info);
        if (refinement) cusolverDnIRSParamsDestroy(refinement);
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
        cudaFree(norms);
        cudaFree(info);
        cudaFree(pivots);
        cudaFree(residual);
        cudaFree(rhs);
        cudaFree(normal);
        cudaFree(vector);
        cudaFree(response);
        cudaFree(uploaded_coupling);
        cudaFree(scratch);
        cudaFree(factor);
        if (stream) cudaStreamDestroy(stream);
    }

    std::size_t matrix_bytes() const { return sizeof(Complex) * std::size_t(rows) * rows; }
    std::size_t coupling_bytes() const { return sizeof(Complex) * std::size_t(rows) * wet; }
    std::size_t vector_bytes() const { return sizeof(Complex) * rows; }
};

constexpr int blas_error = 10000;
constexpr int solver_error = 20000;
constexpr int lapack_error = 30000;
constexpr double refinement_tolerance = 2e-14;

#define CUDA_CHECK(call) do { const auto rc = (call); if (rc != cudaSuccess) return rc; } while (0)
#define BLAS_CHECK(call) do { const auto rc = (call); if (rc != CUBLAS_STATUS_SUCCESS) return blas_error + rc; } while (0)
#define SOLVER_CHECK(call) do { const auto rc = (call); if (rc != CUSOLVER_STATUS_SUCCESS) return solver_error + rc; } while (0)

int check_info(Workspace &ws) {
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, ws.info, sizeof(info), cudaMemcpyDeviceToHost));
    return info ? lapack_error : 0;
}

int factor_matrix(Workspace &ws) {
    SOLVER_CHECK(cusolverDnCreate(&ws.solver));
    CUDA_CHECK(cudaMalloc(&ws.pivots, sizeof(int) * ws.rows));
    CUDA_CHECK(cudaMalloc(&ws.info, sizeof(int)));
    int elements = 0;
    SOLVER_CHECK(cusolverDnZgetrf_bufferSize(ws.solver, ws.rows, ws.rows, ws.factor, ws.rows, &elements));
    WorkBuffer scratch;
    CUDA_CHECK(cudaMalloc(&scratch.values, sizeof(Complex) * std::size_t(elements)));
    SOLVER_CHECK(cusolverDnZgetrf(ws.solver, ws.rows, ws.rows, ws.factor, ws.rows,
                                scratch.values, ws.pivots, ws.info));
    const int rc = check_info(ws);
    ws.factored = rc == 0;
    return rc;
}

int solve_factored(Workspace &ws, cublasOperation_t trans, const Complex *input, Complex *output) {
    if (!ws.factored) return cudaErrorInvalidValue;
    const auto bytes = sizeof(Complex) * ws.rows;
    CUDA_CHECK(cudaMemcpy(ws.vector, input, bytes, cudaMemcpyHostToDevice));
    SOLVER_CHECK(cusolverDnZgetrs(ws.solver, trans, ws.rows, 1,
                                ws.factor, ws.rows, ws.pivots, ws.vector, ws.rows, ws.info));
    const int rc = check_info(ws);
    if (rc) return rc;
    CUDA_CHECK(cudaMemcpy(output, ws.vector, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

int apply_response(cublasHandle_t blas, int rows, int wet, Complex *factor,
                   const Complex *coupling, const Complex *response, bool gemm3m = false) {
    const auto one = make_cuDoubleComplex(1, 0);
    const auto negative_one = make_cuDoubleComplex(-1, 0);
    // The row-major blocks are transposed to cuBLAS. Update the first wet rows of A^T.
    const auto gemm = gemm3m ? cublasZgemm3m : cublasZgemm;
    BLAS_CHECK(gemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, wet, rows, wet,
                    &negative_one, response, wet, coupling, wet, &one, factor, rows));
    return 0;
}

int relative_norm(DirectWorkspace &work, const Complex *residual, const Complex *rhs, double *relative) {
    BLAS_CHECK(cublasSetPointerMode(work.blas, CUBLAS_POINTER_MODE_DEVICE));
    const auto numerator = cublasDznrm2(work.blas, work.rows, residual, 1, work.norms);
    const auto denominator = cublasDznrm2(work.blas, work.rows, rhs, 1, work.norms + 1);
    BLAS_CHECK(cublasSetPointerMode(work.blas, CUBLAS_POINTER_MODE_HOST));
    BLAS_CHECK(numerator);
    BLAS_CHECK(denominator);
    double norms[2]{};
    CUDA_CHECK(cudaMemcpyAsync(norms, work.norms, sizeof(norms), cudaMemcpyDeviceToHost, work.stream));
    CUDA_CHECK(cudaStreamSynchronize(work.stream));
    *relative = norms[0] / std::max(norms[1], std::numeric_limits<double>::min());
    return 0;
}

int schur_defect(DirectWorkspace &work, const Complex *boundary, const Complex *coupling) {
    const auto one = make_cuDoubleComplex(1, 0);
    const auto zero = make_cuDoubleComplex(0, 0);
    const auto minus_one = make_cuDoubleComplex(-1, 0);
    const int n = work.rows;
    const int wet = work.wet;
    // Apply the original FP64 A - C R to p. The retained LU may be FP32,
    // but neither R nor the blocks used for acceptance are rounded to FP32.
    BLAS_CHECK(cublasZgemv(work.blas, CUBLAS_OP_T, wet, wet, &one, work.response,
                          wet, work.vector, 1, &zero, work.normal, 1));
    CUDA_CHECK(cudaMemcpyAsync(work.residual, work.rhs, work.vector_bytes(), cudaMemcpyDeviceToDevice, work.stream));
    BLAS_CHECK(cublasZgemv(work.blas, CUBLAS_OP_T, n, n, &minus_one, boundary,
                          n, work.vector, 1, &one, work.residual, 1));
    BLAS_CHECK(cublasZgemv(work.blas, CUBLAS_OP_T, wet, n, &one, coupling,
                          wet, work.normal, 1, &one, work.residual, 1));
    return 0;
}

int try_refinement(DirectWorkspace &work, const Complex *boundary, const Complex *coupling,
                   const Complex *rhs, cudaMemcpyKind input_kind) {
    const auto one = make_cuDoubleComplex(1, 0);
    const auto zero = make_cuDoubleComplex(0, 0);
    const auto minus_one = make_cuDoubleComplex(-1, 0);
    const Complex *source = boundary;
    if (input_kind == cudaMemcpyHostToDevice) {
        CUDA_CHECK(cudaMemcpyAsync(work.scratch, boundary, work.matrix_bytes(), input_kind, work.stream));
        source = work.scratch;
    }
    // IRS requires column-major input. Reuse one scratch allocation for the host
    // transpose input, refinement, and the double-precision fallback.
    BLAS_CHECK(cublasZgeam(work.blas, CUBLAS_OP_T, CUBLAS_OP_T, work.rows, work.rows, &one,
                          source, work.rows, &zero, source, work.rows, work.factor, work.rows));
    const auto gemm = work.gemm3m ? cublasZgemm3m : cublasZgemm;
    BLAS_CHECK(gemm(work.blas, CUBLAS_OP_T, CUBLAS_OP_T, work.rows, work.wet, work.wet,
                    &minus_one, coupling, work.wet, work.response, work.wet, &one, work.factor, work.rows));
    CUDA_CHECK(cudaMemcpyAsync(work.rhs, rhs, work.vector_bytes(), cudaMemcpyHostToDevice, work.stream));
    CUDA_CHECK(cudaMemsetAsync(work.info, 0, sizeof(cusolver_int_t), work.stream));
    cusolver_int_t iterations = 0;
    const auto status = cusolverDnIRSXgesv(work.solver, work.refinement, work.refinement_info,
                                           work.rows, 1, work.factor, work.rows, work.rhs, work.rows,
                                           work.vector, work.rows, work.scratch, work.refinement_bytes,
                                           &iterations, reinterpret_cast<cusolver_int_t *>(work.info));
    work.refinement_iterations = static_cast<int>(iterations);
    if (status != CUSOLVER_STATUS_SUCCESS || iterations < 0) return 0;

    cusolver_int_t info = 0;
    CUDA_CHECK(cudaMemcpyAsync(&info, work.info, sizeof(info), cudaMemcpyDeviceToHost, work.stream));
    CUDA_CHECK(cudaStreamSynchronize(work.stream));
    if (info) return 0;
    // Check the FP64 Schur system against its unmodified RHS before accepting.
    CUDA_CHECK(cudaMemcpyAsync(work.residual, work.rhs, work.vector_bytes(), cudaMemcpyDeviceToDevice, work.stream));
    BLAS_CHECK(cublasZgemv(work.blas, CUBLAS_OP_N, work.rows, work.rows, &one, work.factor,
                          work.rows, work.vector, 1, &minus_one, work.residual, 1));
    if (const int rc = relative_norm(work, work.residual, work.rhs, &work.refinement_residual)) return rc;
    work.refinement_accepted = std::isfinite(work.refinement_residual)
                               && work.refinement_residual <= refinement_tolerance;
    return 0;
}

#ifdef __CUDACC__

__global__ void store_single(cuComplex *output, const Complex *input, std::size_t count) {
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = cuComplexDoubleToFloat(input[index]);
}

__global__ void add_correction(Complex *output, const cuComplex *correction, int count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = cuCadd(output[index], cuComplexFloatToDouble(correction[index]));
}

int cast_single(DirectWorkspace &work, cuComplex *output, const Complex *input, std::size_t count) {
    constexpr int threads = 256;
    store_single<<<(count + threads - 1) / threads, threads, 0, work.stream>>>(output, input, count);
    return cudaGetLastError();
}

int prepare_single_factor(DirectWorkspace &work, const Complex *boundary, const Complex *coupling) {
    const int n = work.rows;
    const int wet = work.wet;
    auto *single_factor = reinterpret_cast<cuComplex *>(work.factor);
    auto *single_coupling = reinterpret_cast<cuComplex *>(work.scratch);
    auto *single_response = single_coupling + std::size_t(n) * wet;

    if (const int rc = cast_single(work, single_factor, boundary, std::size_t(n) * n)) return rc;
    if (const int rc = cast_single(work, single_coupling, coupling, std::size_t(n) * wet)) return rc;
    if (const int rc = cast_single(work, single_response, work.response, std::size_t(wet) * wet)) return rc;
    const auto single_one = make_cuComplex(1, 0);
    const auto single_minus_one = make_cuComplex(-1, 0);
    BLAS_CHECK(cublasCgemm(work.blas, CUBLAS_OP_N, CUBLAS_OP_N, wet, n, wet,
                          &single_minus_one, single_response, wet, single_coupling, wet,
                          &single_one, single_factor, n));
    // The product inputs are finished. Their storage is reused as LU scratch.
    auto *single_scratch = reinterpret_cast<cuComplex *>(work.scratch);
    SOLVER_CHECK(cusolverDnCgetrf(work.solver, n, n, single_factor, n, single_scratch, work.pivots, work.info));
    work.factor_precision = FactorPrecision::single;
    return 0;
}

int refine_single_factor(DirectWorkspace &work, const Complex *boundary, const Complex *coupling,
                         const Complex *rhs, bool &accepted, int &iterations, double &relative) {
    const int n = work.rows;
    auto *single_factor = reinterpret_cast<cuComplex *>(work.factor);
    auto *correction = reinterpret_cast<cuComplex *>(work.normal);
    CUDA_CHECK(cudaMemcpyAsync(work.rhs, rhs, work.vector_bytes(), cudaMemcpyHostToDevice, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(work.residual, work.rhs, work.vector_bytes(), cudaMemcpyDeviceToDevice, work.stream));
    CUDA_CHECK(cudaMemsetAsync(work.vector, 0, work.vector_bytes(), work.stream));
    double previous = std::numeric_limits<double>::infinity();
    constexpr int max_solves = 5;
    constexpr int threads = 256;

    for (int i = 0; i < max_solves; ++i) {
        if (const int rc = cast_single(work, correction, work.residual, n)) return rc;
        SOLVER_CHECK(cusolverDnCgetrs(work.solver, CUBLAS_OP_T, n, 1, single_factor, n,
                                     work.pivots, correction, n, work.info + 1));
        add_correction<<<(n + threads - 1) / threads, threads, 0, work.stream>>>(work.vector, correction, n);
        CUDA_CHECK(cudaGetLastError());

        if (const int rc = schur_defect(work, boundary, coupling)) return rc;
        int info[2]{};
        CUDA_CHECK(cudaMemcpyAsync(info, work.info, sizeof(info), cudaMemcpyDeviceToHost, work.stream));
        if (const int rc = relative_norm(work, work.residual, work.rhs, &relative)) return rc;
        iterations = i;
        if (info[0] || info[1] || !std::isfinite(relative)) break;
        if (relative <= refinement_tolerance) {
            accepted = true;
            return 0;
        }
        if (relative >= previous) break;
        previous = relative;
    }
    return 0;
}

#endif

int prepare_double_factor(DirectWorkspace &work, const Complex *boundary, const Complex *coupling,
                          cudaMemcpyKind input_kind) {
    work.factor_precision = FactorPrecision::none;
    CUDA_CHECK(cudaMemcpyAsync(work.factor, boundary, work.matrix_bytes(), input_kind, work.stream));
    if (const int rc = apply_response(work.blas, work.rows, work.wet, work.factor, coupling, work.response,
                                      work.gemm3m && !work.refinement && !work.fast_refinement)) {
        return rc;
    }
    SOLVER_CHECK(cusolverDnZgetrf(work.solver, work.rows, work.rows, work.factor, work.rows,
                                work.scratch, work.pivots, work.info));
    work.factor_precision = FactorPrecision::double_precision;
    return 0;
}

int solve_double_factor(DirectWorkspace &work, Complex *rhs) {
    // Distinct status entries retain a singular-factorization error after the solve is enqueued.
    SOLVER_CHECK(cusolverDnZgetrs(work.solver, CUBLAS_OP_T, work.rows, 1, work.factor, work.rows,
                                work.pivots, rhs, work.rows, work.info + 1));
    return 0;
}

int direct_solve(DirectWorkspace &work, const Complex *boundary, const Complex *coupling,
                 const Complex *response, const Complex *rhs, Complex *solution,
                 cudaMemcpyKind input_kind) {
    work.factor_precision = FactorPrecision::none;
    work.prepared_boundary = nullptr;
    work.prepared_coupling = nullptr;
    CUDA_CHECK(cudaMemcpyAsync(work.response, response, sizeof(Complex) * std::size_t(work.wet) * work.wet,
                               cudaMemcpyHostToDevice, work.stream));
    if (work.refinement || work.fast_refinement) {
        work.refinement_iterations = 0;
        work.refinement_accepted = false;
        work.refinement_residual = std::numeric_limits<double>::quiet_NaN();
#ifdef __CUDACC__
        if (work.fast_refinement) {
            if (input_kind != cudaMemcpyDeviceToDevice) return cudaErrorInvalidValue;
            if (const int rc = prepare_single_factor(work, boundary, coupling)) return rc;
            if (const int rc = refine_single_factor(work, boundary, coupling, rhs, work.refinement_accepted,
                                                    work.refinement_iterations, work.refinement_residual)) return rc;
        } else
#endif
        if (const int rc = try_refinement(work, boundary, coupling, rhs, input_kind)) return rc;
        if (work.refinement_accepted) {
            CUDA_CHECK(cudaMemcpyAsync(solution, work.vector, work.vector_bytes(), cudaMemcpyDeviceToHost, work.stream));
            return cudaStreamSynchronize(work.stream);
        }
    }
    if (const int rc = prepare_double_factor(work, boundary, coupling, input_kind)) return rc;
    CUDA_CHECK(cudaMemcpyAsync(work.vector, rhs, work.vector_bytes(), cudaMemcpyHostToDevice, work.stream));
    if (const int rc = solve_double_factor(work, work.vector)) return rc;
    int info[2]{};
    CUDA_CHECK(cudaMemcpyAsync(solution, work.vector, work.vector_bytes(), cudaMemcpyDeviceToHost, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(info, work.info, sizeof(info), cudaMemcpyDeviceToHost, work.stream));
    CUDA_CHECK(cudaStreamSynchronize(work.stream));
    return info[0] || info[1] ? lapack_error : 0;
}

int refine_double_factor(DirectWorkspace &work, const Complex *boundary, const Complex *coupling,
                         const Complex *rhs, double &relative) {
    const auto one = make_cuDoubleComplex(1, 0);
    const int n = work.rows;
    CUDA_CHECK(cudaMemcpyAsync(work.rhs, rhs, work.vector_bytes(), cudaMemcpyHostToDevice, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(work.vector, work.rhs, work.vector_bytes(), cudaMemcpyDeviceToDevice, work.stream));
    if (const int rc = solve_double_factor(work, work.vector)) return rc;
    double previous = std::numeric_limits<double>::infinity();
    constexpr int max_refinements = 2;
    for (int i = 0; i <= max_refinements; ++i) {
        if (const int rc = schur_defect(work, boundary, coupling)) return rc;
        int info[2]{};
        CUDA_CHECK(cudaMemcpyAsync(info, work.info, sizeof(info), cudaMemcpyDeviceToHost, work.stream));
        if (const int rc = relative_norm(work, work.residual, work.rhs, &relative)) return rc;
        if (info[0] || info[1] || !std::isfinite(relative)) return lapack_error;
        if (relative <= refinement_tolerance) return 0;
        if (i == max_refinements || relative >= previous) break;
        previous = relative;
        if (const int rc = solve_double_factor(work, work.residual)) return rc;
        BLAS_CHECK(cublasZaxpy(work.blas, n, &one, work.residual, 1, work.vector, 1));
    }
    return lapack_error + 1;
}

int solve_prepared(DirectWorkspace &work, const Complex *boundary, const Complex *coupling,
                   int rhs_count, const Complex *rhs, Complex *solution, double *residuals) {
    if (rhs_count < 1 || !boundary || !coupling || boundary != work.prepared_boundary
        || coupling != work.prepared_coupling) return cudaErrorInvalidValue;
    for (int column = 0; column < rhs_count; ++column) {
        const auto offset = std::size_t(column) * work.rows;
        bool accepted = false;
#ifdef __CUDACC__
        if (work.factor_precision == FactorPrecision::single) {
            int iterations = 0;
            if (const int rc = refine_single_factor(work, boundary, coupling, rhs + offset, accepted,
                                                    iterations, residuals[column])) return rc;
        }
#endif
        if (!accepted) {
            if (work.factor_precision != FactorPrecision::double_precision) {
                // If this RHS cannot use the single-precision LU, retain one
                // double-precision replacement for this and subsequent RHS.
                if (const int rc = prepare_double_factor(work, boundary, coupling, cudaMemcpyDeviceToDevice)) {
                    return rc;
                }
            }
            if (const int rc = refine_double_factor(work, boundary, coupling, rhs + offset, residuals[column])) {
                return rc;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(solution + offset, work.vector, work.vector_bytes(),
                                   cudaMemcpyDeviceToHost, work.stream));
    }
    return cudaStreamSynchronize(work.stream);
}

#ifdef __CUDACC__

constexpr int batch_threads = 256;
constexpr int batch_active = 0;
constexpr int batch_certified = 1;
constexpr int batch_failed = 2;

int reserve_batch(DirectWorkspace &work, int count) {
    if (work.batch && work.batch->capacity >= count) return 0;
    auto buffers = std::unique_ptr<BatchWorkspace>(new (std::nothrow) BatchWorkspace);
    if (!buffers) return cudaErrorMemoryAllocation;
    const auto bytes = work.vector_bytes() * count;
    CUDA_CHECK(cudaMalloc(&buffers->rhs, bytes));
    CUDA_CHECK(cudaMalloc(&buffers->solution, bytes));
    CUDA_CHECK(cudaMalloc(&buffers->defect, bytes));
    CUDA_CHECK(cudaMalloc(&buffers->normal, sizeof(Complex) * std::size_t(work.wet) * count));
    CUDA_CHECK(cudaMalloc(&buffers->correction, sizeof(cuComplex) * std::size_t(work.rows) * count));
    CUDA_CHECK(cudaMalloc(&buffers->relative, sizeof(double) * count));
    CUDA_CHECK(cudaMalloc(&buffers->state, sizeof(int) * count));
    CUDA_CHECK(cudaMalloc(&buffers->pending, sizeof(int)));
    buffers->capacity = count;
    work.batch = std::move(buffers);
    return 0;
}

__global__ void batch_single_rhs(cuComplex *correction, const Complex *defect, const int *state,
                                 std::size_t size, int rows) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        correction[i] = state[i / rows] == batch_active ? cuComplexDoubleToFloat(defect[i]) : make_cuComplex(0, 0);
    }
}

__global__ void batch_add_single(Complex *solution, const cuComplex *correction, const int *state,
                                 std::size_t size, int rows) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size && state[i / rows] == batch_active) {
        solution[i] = cuCadd(solution[i], cuComplexFloatToDouble(correction[i]));
    }
}

__global__ void batch_mask_double(Complex *defect, const int *state, std::size_t size, int rows) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size && state[i / rows] != batch_active) defect[i] = make_cuDoubleComplex(0, 0);
}

__global__ void batch_add_double(Complex *solution, const Complex *correction, const int *state,
                                 std::size_t size, int rows) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size && state[i / rows] == batch_active) solution[i] = cuCadd(solution[i], correction[i]);
}

__device__ double finite_scale(Complex value) {
    return isfinite(value.x) && isfinite(value.y) ? fmax(fabs(value.x), fabs(value.y)) : INFINITY;
}

__device__ double scaled_square(Complex value, double scale) {
    if (scale == 0.) return 0.;
    const double x = value.x / scale, y = value.y / scale;
    return x * x + y * y;
}

__global__ void certify_batch(const Complex *defect, const Complex *rhs, int rows, const int *info,
                              int iteration, int max_refinements, double *relative, int *state, int *pending) {
    const int column = blockIdx.x;
    if (state[column] != batch_active) return;
    if (info[0] || info[1]) {
        if (!threadIdx.x) state[column] = batch_failed;
        return;
    }
    const auto offset = std::size_t(column) * rows;
    const int thread = threadIdx.x;
    __shared__ double defect_norm[batch_threads];
    __shared__ double rhs_norm[batch_threads];
    double defect_scale = 0., rhs_scale = 0.;
    for (int i = thread; i < rows; i += batch_threads) {
        defect_scale = fmax(defect_scale, finite_scale(defect[offset + i]));
        rhs_scale = fmax(rhs_scale, finite_scale(rhs[offset + i]));
    }
    defect_norm[thread] = defect_scale;
    rhs_norm[thread] = rhs_scale;
    __syncthreads();
    for (int stride = batch_threads / 2; stride; stride /= 2) {
        if (thread < stride) {
            defect_norm[thread] = fmax(defect_norm[thread], defect_norm[thread + stride]);
            rhs_norm[thread] = fmax(rhs_norm[thread], rhs_norm[thread + stride]);
        }
        __syncthreads();
    }
    defect_scale = defect_norm[0];
    rhs_scale = rhs_norm[0];
    __syncthreads();
    if (!isfinite(defect_scale) || !isfinite(rhs_scale)) {
        if (!thread) state[column] = batch_failed;
        return;
    }
    double defect_sum = 0., rhs_sum = 0.;
    for (int i = thread; i < rows; i += batch_threads) {
        defect_sum += scaled_square(defect[offset + i], defect_scale);
        rhs_sum += scaled_square(rhs[offset + i], rhs_scale);
    }
    defect_norm[thread] = defect_sum;
    rhs_norm[thread] = rhs_sum;
    __syncthreads();
    for (int stride = batch_threads / 2; stride; stride /= 2) {
        if (thread < stride) {
            defect_norm[thread] += defect_norm[thread + stride];
            rhs_norm[thread] += rhs_norm[thread + stride];
        }
        __syncthreads();
    }
    if (!thread) {
        const double denominator = rhs_scale * sqrt(rhs_norm[0]);
        const double numerator = defect_scale * sqrt(defect_norm[0]);
        const double error = numerator / fmax(denominator, DBL_MIN);
        if (!isfinite(error) || !isfinite(denominator)) {
            state[column] = batch_failed;
        } else if (error <= refinement_tolerance) {
            state[column] = batch_certified;
        } else if (iteration == max_refinements || (iteration > 0 && error >= relative[column])) {
            state[column] = batch_failed;
        } else {
            atomicExch(pending, 1);
        }
        relative[column] = error;
    }
}

int schur_defect_batch(DirectWorkspace &work, const Complex *boundary, const Complex *coupling, int count) {
    auto &batch = *work.batch;
    const auto one = make_cuDoubleComplex(1, 0);
    const auto zero = make_cuDoubleComplex(0, 0);
    const auto minus_one = make_cuDoubleComplex(-1, 0);
    const int n = work.rows, wet = work.wet;
    // Certify each column using the original FP64 operators and unmodified right-hand side.
    BLAS_CHECK(cublasZgemm(work.blas, CUBLAS_OP_T, CUBLAS_OP_N, wet, count, wet, &one, work.response,
                          wet, batch.solution, n, &zero, batch.normal, wet));
    CUDA_CHECK(cudaMemcpyAsync(batch.defect, batch.rhs, work.vector_bytes() * count,
                               cudaMemcpyDeviceToDevice, work.stream));
    BLAS_CHECK(cublasZgemm(work.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, count, n, &minus_one, boundary,
                          n, batch.solution, n, &one, batch.defect, n));
    BLAS_CHECK(cublasZgemm(work.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, count, wet, &one, coupling,
                          wet, batch.normal, wet, &one, batch.defect, n));
    return 0;
}

int solve_prepared_batch(DirectWorkspace &work, const Complex *boundary, const Complex *coupling, int count,
                         const Complex *rhs, const Complex *normal, Complex *solution, double *residuals) {
    if (count < 1 || !boundary || !coupling || boundary != work.prepared_boundary
        || coupling != work.prepared_coupling) return cudaErrorInvalidValue;
    if (const int rc = reserve_batch(work, count)) return rc;
    auto &batch = *work.batch;
    const int n = work.rows, wet = work.wet;
    const auto elements = std::size_t(n) * count;
    const auto bytes = work.vector_bytes() * count;
    const auto blocks = (elements + batch_threads - 1) / batch_threads;
    CUDA_CHECK(cudaMemcpyAsync(batch.rhs, rhs, bytes, cudaMemcpyHostToDevice, work.stream));
    if (normal) {
        const auto one = make_cuDoubleComplex(1, 0);
        CUDA_CHECK(cudaMemcpyAsync(batch.normal, normal, sizeof(Complex) * std::size_t(wet) * count,
                                   cudaMemcpyHostToDevice, work.stream));
        BLAS_CHECK(cublasZgemm(work.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, count, wet, &one, coupling,
                              wet, batch.normal, wet, &one, batch.rhs, n));
    }
    CUDA_CHECK(cudaMemsetAsync(batch.state, 0, sizeof(int) * count, work.stream));
    CUDA_CHECK(cudaMemsetAsync(batch.relative, 0, sizeof(double) * count, work.stream));

    const bool single = work.factor_precision == FactorPrecision::single;
    if (!single && work.factor_precision != FactorPrecision::double_precision) {
        if (const int rc = prepare_double_factor(work, boundary, coupling, cudaMemcpyDeviceToDevice)) return rc;
    }
    if (single) {
        CUDA_CHECK(cudaMemsetAsync(batch.solution, 0, bytes, work.stream));
        CUDA_CHECK(cudaMemcpyAsync(batch.defect, batch.rhs, bytes, cudaMemcpyDeviceToDevice, work.stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(batch.solution, batch.rhs, bytes, cudaMemcpyDeviceToDevice, work.stream));
        SOLVER_CHECK(cusolverDnZgetrs(work.solver, CUBLAS_OP_T, n, count, work.factor, n,
                                     work.pivots, batch.solution, n, work.info + 1));
    }

    const int max_refinements = single ? 4 : 2;
    for (int iteration = 0; iteration <= max_refinements; ++iteration) {
        if (single) {
            batch_single_rhs<<<blocks, batch_threads, 0, work.stream>>>(
                batch.correction, batch.defect, batch.state, elements, n);
            CUDA_CHECK(cudaGetLastError());
            SOLVER_CHECK(cusolverDnCgetrs(work.solver, CUBLAS_OP_T, n, count,
                                         reinterpret_cast<cuComplex *>(work.factor), n,
                                         work.pivots, batch.correction, n, work.info + 1));
            batch_add_single<<<blocks, batch_threads, 0, work.stream>>>(
                batch.solution, batch.correction, batch.state, elements, n);
            CUDA_CHECK(cudaGetLastError());
        } else if (iteration > 0) {
            batch_mask_double<<<blocks, batch_threads, 0, work.stream>>>(batch.defect, batch.state, elements, n);
            CUDA_CHECK(cudaGetLastError());
            SOLVER_CHECK(cusolverDnZgetrs(work.solver, CUBLAS_OP_T, n, count, work.factor, n,
                                         work.pivots, batch.defect, n, work.info + 1));
            batch_add_double<<<blocks, batch_threads, 0, work.stream>>>(
                batch.solution, batch.defect, batch.state, elements, n);
            CUDA_CHECK(cudaGetLastError());
        }
        if (const int rc = schur_defect_batch(work, boundary, coupling, count)) return rc;
        CUDA_CHECK(cudaMemsetAsync(batch.pending, 0, sizeof(int), work.stream));
        certify_batch<<<count, batch_threads, 0, work.stream>>>(
            batch.defect, batch.rhs, n, work.info, iteration, max_refinements,
            batch.relative, batch.state, batch.pending);
        CUDA_CHECK(cudaGetLastError());
        int pending = 0;
        CUDA_CHECK(cudaMemcpyAsync(&pending, batch.pending, sizeof(pending), cudaMemcpyDeviceToHost, work.stream));
        CUDA_CHECK(cudaStreamSynchronize(work.stream));
        if (!pending) break;
    }

    const std::unique_ptr<int[]> state(new (std::nothrow) int[count]);
    if (!state) return cudaErrorMemoryAllocation;
    CUDA_CHECK(cudaMemcpyAsync(state.get(), batch.state, sizeof(int) * count, cudaMemcpyDeviceToHost, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(residuals, batch.relative, sizeof(double) * count, cudaMemcpyDeviceToHost, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(solution, batch.solution, bytes, cudaMemcpyDeviceToHost, work.stream));
    CUDA_CHECK(cudaStreamSynchronize(work.stream));

    // Keep certified columns and use the existing individually checked path for any exception.
    std::unique_ptr<Complex[]> original_rhs;
    for (int column = 0; column < count; ++column) {
        if (state[column] == batch_certified) continue;
        const auto offset = std::size_t(column) * n;
        const Complex *column_rhs = rhs + offset;
        if (normal) {
            if (!original_rhs) original_rhs.reset(new (std::nothrow) Complex[n]);
            if (!original_rhs) return cudaErrorMemoryAllocation;
            CUDA_CHECK(cudaMemcpyAsync(original_rhs.get(), batch.rhs + offset, work.vector_bytes(),
                                       cudaMemcpyDeviceToHost, work.stream));
            CUDA_CHECK(cudaStreamSynchronize(work.stream));
            column_rhs = original_rhs.get();
        }
        if (const int rc = solve_prepared(work, boundary, coupling, 1, column_rhs,
                                          solution + offset, residuals + column)) return rc;
    }
    return 0;
}

#endif

int coupling_product(DirectWorkspace &work, const Complex *coupling, int rhs_count,
                     const Complex *normal, Complex *product) {
    if (!coupling || rhs_count < 1) return cudaErrorInvalidValue;
    const auto one = make_cuDoubleComplex(1, 0);
    const auto zero = make_cuDoubleComplex(0, 0);
    const auto input_count = std::size_t(work.wet) * rhs_count;
    const auto input_bytes = sizeof(Complex) * input_count;
    const auto output_bytes = work.vector_bytes() * rhs_count;
    if (rhs_count == 1) {
        CUDA_CHECK(cudaMemcpyAsync(work.normal, normal, input_bytes, cudaMemcpyHostToDevice, work.stream));
        BLAS_CHECK(cublasZgemv(work.blas, CUBLAS_OP_T, work.wet, work.rows, &one, coupling, work.wet,
                              work.normal, 1, &zero, work.vector, 1));
        CUDA_CHECK(cudaMemcpyAsync(product, work.vector, output_bytes, cudaMemcpyDeviceToHost, work.stream));
        return cudaStreamSynchronize(work.stream);
    }
    WorkBuffer temporary;
    Complex *input = work.scratch;
    if (work.scratch_bytes < input_bytes + output_bytes) {
        CUDA_CHECK(cudaMalloc(&temporary.values, input_bytes + output_bytes));
        input = temporary.values;
    }
    auto *output = input + input_count;
    CUDA_CHECK(cudaMemcpyAsync(input, normal, input_bytes, cudaMemcpyHostToDevice, work.stream));
    BLAS_CHECK(cublasZgemm(work.blas, CUBLAS_OP_T, CUBLAS_OP_N, work.rows, rhs_count, work.wet,
                          &one, coupling, work.wet, input, work.wet, &zero, output, work.rows));
    CUDA_CHECK(cudaMemcpyAsync(product, output, output_bytes, cudaMemcpyDeviceToHost, work.stream));
    return cudaStreamSynchronize(work.stream);
}

}  // namespace

extern "C" int planar_water_create(int device, int rows, int wet, const Complex *boundary,
                                   const Complex *coupling, void **result) {
    *result = nullptr;
    if (rows < 1 || wet < 1 || wet > rows) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaSetDevice(device));
    std::unique_ptr<Workspace> ws(new (std::nothrow) Workspace);
    if (!ws) return cudaErrorMemoryAllocation;
    ws->device = device;
    ws->rows = rows;
    ws->wet = wet;
    const auto full_bytes = sizeof(Complex) * std::size_t(rows) * rows;
    const auto wet_bytes = sizeof(Complex) * std::size_t(rows) * wet;
    CUDA_CHECK(cudaMalloc(&ws->boundary, full_bytes));
    CUDA_CHECK(cudaMalloc(&ws->coupling, wet_bytes));
    CUDA_CHECK(cudaMalloc(&ws->vector, sizeof(Complex) * rows));
    CUDA_CHECK(cudaMalloc(&ws->normal, sizeof(Complex) * wet));
    CUDA_CHECK(cudaMalloc(&ws->result, sizeof(Complex) * rows));
    CUDA_CHECK(cudaMemcpy(ws->boundary, boundary, full_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws->coupling, coupling, wet_bytes, cudaMemcpyHostToDevice));
    BLAS_CHECK(cublasCreate(&ws->blas));
    *result = ws.release();
    return 0;
}

extern "C" void planar_water_destroy(void *workspace) {
    delete static_cast<Workspace *>(workspace);
}

extern "C" int planar_water_action(void *workspace, const Complex *vector,
                                   const Complex *normal, Complex *result) {
    auto &ws = *static_cast<Workspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(ws.device));
    CUDA_CHECK(cudaMemcpy(ws.vector, vector, sizeof(Complex) * ws.rows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ws.normal, normal, sizeof(Complex) * ws.wet, cudaMemcpyHostToDevice));
    const auto one = make_cuDoubleComplex(1, 0);
    const auto zero = make_cuDoubleComplex(0, 0);
    const auto negative_one = make_cuDoubleComplex(-1, 0);
    // Input blocks are row-major. Transpose the cuBLAS column-major view
    // without conjugation to preserve the original complex operator.
    BLAS_CHECK(cublasZgemv(ws.blas, CUBLAS_OP_T, ws.rows, ws.rows, &one, ws.boundary,
                          ws.rows, ws.vector, 1, &zero, ws.result, 1));
    BLAS_CHECK(cublasZgemv(ws.blas, CUBLAS_OP_T, ws.wet, ws.rows, &negative_one, ws.coupling,
                          ws.wet, ws.normal, 1, &one, ws.result, 1));
    CUDA_CHECK(cudaMemcpy(result, ws.result, sizeof(Complex) * ws.rows, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int planar_water_factor(void *workspace, const Complex *preconditioner) {
    auto &ws = *static_cast<Workspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(ws.device));
    if (ws.factor) return cudaErrorInvalidValue;
    const auto bytes = sizeof(Complex) * std::size_t(ws.rows) * ws.rows;
    CUDA_CHECK(cudaMalloc(&ws.factor, bytes));
    // Preconditioner is already column-major on the host.
    CUDA_CHECK(cudaMemcpy(ws.factor, preconditioner, bytes, cudaMemcpyHostToDevice));
    return factor_matrix(ws);
}

extern "C" int planar_water_precondition(void *workspace, const Complex *input, Complex *output) {
    auto &ws = *static_cast<Workspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(ws.device));
    return solve_factored(ws, CUBLAS_OP_N, input, output);
}

extern "C" int planar_water_direct(int device, int rows, int wet, const Complex *boundary,
                                   const Complex *coupling, const Complex *response,
                                   const Complex *rhs, Complex *solution) {
    if (rows < 1 || wet < 1 || wet > rows) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaSetDevice(device));
    Workspace ws;
    ws.device = device;
    ws.rows = rows;
    ws.wet = wet;
    const auto matrix_bytes = sizeof(Complex) * std::size_t(rows) * rows;
    CUDA_CHECK(cudaMalloc(&ws.factor, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(ws.factor, boundary, matrix_bytes, cudaMemcpyHostToDevice));
    {
        WorkBuffer device_coupling, device_response;
        const auto coupling_bytes = sizeof(Complex) * std::size_t(rows) * wet;
        const auto response_bytes = sizeof(Complex) * std::size_t(wet) * wet;
        CUDA_CHECK(cudaMalloc(&device_coupling.values, coupling_bytes));
        CUDA_CHECK(cudaMalloc(&device_response.values, response_bytes));
        CUDA_CHECK(cudaMemcpy(device_coupling.values, coupling, coupling_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_response.values, response, response_bytes, cudaMemcpyHostToDevice));
        BLAS_CHECK(cublasCreate(&ws.blas));
        if (const int rc = apply_response(ws.blas, rows, wet, ws.factor,
                                          device_coupling.values, device_response.values)) return rc;
    }
    if (const int rc = factor_matrix(ws)) return rc;
    CUDA_CHECK(cudaMalloc(&ws.vector, sizeof(Complex) * rows));
    return solve_factored(ws, CUBLAS_OP_T, rhs, solution);
}

namespace {

int create_direct_workspace(int device, int rows, int wet, bool mixed_precision, bool fast_refinement, void **result) {
    *result = nullptr;
    if (rows < 1 || wet < 1 || wet > rows) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaSetDevice(device));
    std::unique_ptr<DirectWorkspace> work(new (std::nothrow) DirectWorkspace(device, rows, wet));
    if (!work) return cudaErrorMemoryAllocation;
    work->fast_refinement = fast_refinement;
    CUDA_CHECK(cudaStreamCreateWithFlags(&work->stream, cudaStreamNonBlocking));
    BLAS_CHECK(cublasCreate(&work->blas));
    BLAS_CHECK(cublasSetStream(work->blas, work->stream));
    SOLVER_CHECK(cusolverDnCreate(&work->solver));
    SOLVER_CHECK(cusolverDnSetStream(work->solver, work->stream));
    CUDA_CHECK(cudaMalloc(&work->factor, work->matrix_bytes()));
    CUDA_CHECK(cudaMalloc(&work->response, sizeof(Complex) * std::size_t(wet) * wet));
    CUDA_CHECK(cudaMalloc(&work->vector, work->vector_bytes()));
    const auto normal_bytes = sizeof(Complex) * wet;
    CUDA_CHECK(cudaMalloc(&work->normal, fast_refinement ? std::max(normal_bytes, sizeof(cuComplex) * rows) : normal_bytes));
    CUDA_CHECK(cudaMalloc(&work->rhs, work->vector_bytes()));
    CUDA_CHECK(cudaMalloc(&work->residual, work->vector_bytes()));
    CUDA_CHECK(cudaMalloc(&work->pivots, sizeof(int) * rows));
    CUDA_CHECK(cudaMalloc(&work->info, std::max(2 * sizeof(int), sizeof(cusolver_int_t))));
    CUDA_CHECK(cudaMalloc(&work->norms, 2 * sizeof(double)));
    int elements = 0;
    SOLVER_CHECK(cusolverDnZgetrf_bufferSize(work->solver, rows, rows, work->factor, rows, &elements));
    std::size_t scratch_bytes = sizeof(Complex) * std::size_t(elements);
    if (fast_refinement) {
        int single_elements = 0;
        SOLVER_CHECK(cusolverDnCgetrf_bufferSize(work->solver, rows, rows,
                                               reinterpret_cast<cuComplex *>(work->factor), rows, &single_elements));
        const auto single_scratch_bytes = sizeof(cuComplex) * std::size_t(single_elements);
        const auto product_inputs_bytes = sizeof(cuComplex) * std::size_t(wet) * (std::size_t(rows) + wet);
        scratch_bytes = std::max({scratch_bytes, single_scratch_bytes, product_inputs_bytes});
    }
    if (mixed_precision) {
        SOLVER_CHECK(cusolverDnIRSParamsCreate(&work->refinement));
        SOLVER_CHECK(cusolverDnIRSInfosCreate(&work->refinement_info));
        SOLVER_CHECK(cusolverDnIRSParamsSetSolverPrecisions(work->refinement, CUSOLVER_C_64F, CUSOLVER_C_32F));
        SOLVER_CHECK(cusolverDnIRSParamsSetRefinementSolver(work->refinement, CUSOLVER_IRS_REFINE_CLASSICAL));
        SOLVER_CHECK(cusolverDnIRSParamsSetMaxIters(work->refinement, 8));
        SOLVER_CHECK(cusolverDnIRSParamsDisableFallback(work->refinement));
        SOLVER_CHECK(cusolverDnIRSXgesv_bufferSize(work->solver, work->refinement, rows, 1, &work->refinement_bytes));
        scratch_bytes = std::max({scratch_bytes, work->matrix_bytes(), work->refinement_bytes});
    }
    CUDA_CHECK(cudaMalloc(&work->scratch, scratch_bytes));
    work->scratch_bytes = scratch_bytes;
    *result = work.release();
    return 0;
}

}  // namespace

extern "C" int planar_water_direct_create(int device, int rows, int wet, void **result) {
    return create_direct_workspace(device, rows, wet, false, false, result);
}

extern "C" int planar_water_direct_create_mixed(int device, int rows, int wet, void **result) {
    return create_direct_workspace(device, rows, wet, true, false, result);
}

#ifdef __CUDACC__
extern "C" int planar_water_direct_create_fast(int device, int rows, int wet, void **result) {
    return create_direct_workspace(device, rows, wet, false, true, result);
}
#endif

extern "C" void planar_water_direct_use_gemm3m(void *workspace) {
    static_cast<DirectWorkspace *>(workspace)->gemm3m = true;
}

extern "C" void planar_water_direct_refinement_stats(void *workspace, int *accepted, int *iterations, double *residual) {
    const auto &work = *static_cast<DirectWorkspace *>(workspace);
    *accepted = work.refinement_accepted;
    *iterations = work.refinement_iterations;
    *residual = work.refinement_residual;
}

extern "C" void planar_water_direct_destroy(void *workspace) {
    delete static_cast<DirectWorkspace *>(workspace);
}

extern "C" int planar_water_direct_solve(void *workspace, const Complex *boundary, const Complex *coupling,
                                         const Complex *response, const Complex *rhs, Complex *solution) {
    auto &work = *static_cast<DirectWorkspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(work.device));
    if (!work.uploaded_coupling) {
        CUDA_CHECK(cudaMalloc(&work.uploaded_coupling, work.coupling_bytes()));
    }
    CUDA_CHECK(cudaMemcpyAsync(work.uploaded_coupling, coupling, work.coupling_bytes(),
                               cudaMemcpyHostToDevice, work.stream));
    const int rc = direct_solve(work, boundary, work.uploaded_coupling, response, rhs, solution,
                                cudaMemcpyHostToDevice);
    if (rc) cudaStreamSynchronize(work.stream);
    return rc;
}

extern "C" int planar_water_direct_resident(void *workspace, const Complex *boundary, const Complex *coupling,
                                            const Complex *response, const Complex *rhs, Complex *solution,
                                            cudaEvent_t ready_event) {
    auto &work = *static_cast<DirectWorkspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(work.device));
    if (ready_event) CUDA_CHECK(cudaStreamWaitEvent(work.stream, ready_event, 0));
    const int rc = direct_solve(work, boundary, coupling, response, rhs, solution, cudaMemcpyDeviceToDevice);
    if (rc) {
        cudaStreamSynchronize(work.stream);
    } else {
        work.prepared_boundary = boundary;
        work.prepared_coupling = coupling;
    }
    return rc;
}

extern "C" int planar_water_direct_solve_prepared(void *workspace, const Complex *boundary, const Complex *coupling,
                                                  int rhs_count, const Complex *rhs, Complex *solution,
                                                  double *relative_residuals) {
    auto &work = *static_cast<DirectWorkspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(work.device));
    const int rc = solve_prepared(work, boundary, coupling, rhs_count, rhs, solution, relative_residuals);
    if (rc) cudaStreamSynchronize(work.stream);
    return rc;
}

#ifdef __CUDACC__
extern "C" int planar_water_direct_solve_prepared_batch(void *workspace, const Complex *boundary,
                                                         const Complex *coupling, int rhs_count,
                                                         const Complex *rhs, const Complex *normal,
                                                         Complex *solution, double *relative_residuals) {
    auto &work = *static_cast<DirectWorkspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(work.device));
    const int rc = solve_prepared_batch(work, boundary, coupling, rhs_count, rhs, normal,
                                        solution, relative_residuals);
    if (rc) cudaStreamSynchronize(work.stream);
    return rc;
}
#endif

extern "C" int planar_water_direct_coupling(void *workspace, const Complex *coupling, int rhs_count,
                                            const Complex *normal, Complex *product, cudaEvent_t ready_event) {
    auto &work = *static_cast<DirectWorkspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(work.device));
    if (ready_event) CUDA_CHECK(cudaStreamWaitEvent(work.stream, ready_event, 0));
    const int rc = coupling_product(work, coupling, rhs_count, normal, product);
    if (rc) cudaStreamSynchronize(work.stream);
    return rc;
}

namespace {

int direct_residual(DirectWorkspace &work, const Complex *boundary, const Complex *coupling,
                    const Complex *solution, const Complex *normal, const Complex *rhs, double *relative) {
    const auto one = make_cuDoubleComplex(1, 0);
    const auto zero = make_cuDoubleComplex(0, 0);
    const auto negative_one = make_cuDoubleComplex(-1, 0);
    CUDA_CHECK(cudaMemcpyAsync(work.vector, solution, work.vector_bytes(), cudaMemcpyHostToDevice, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(work.normal, normal, sizeof(Complex) * work.wet, cudaMemcpyHostToDevice, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(work.rhs, rhs, work.vector_bytes(), cudaMemcpyHostToDevice, work.stream));
    BLAS_CHECK(cublasZgemv(work.blas, CUBLAS_OP_T, work.rows, work.rows, &one, boundary,
                          work.rows, work.vector, 1, &zero, work.residual, 1));
    BLAS_CHECK(cublasZgemv(work.blas, CUBLAS_OP_T, work.wet, work.rows, &negative_one, coupling,
                          work.wet, work.normal, 1, &one, work.residual, 1));
    BLAS_CHECK(cublasZaxpy(work.blas, work.rows, &negative_one, work.rhs, 1, work.residual, 1));
    return relative_norm(work, work.residual, work.rhs, relative);
}

}  // namespace

extern "C" int planar_water_direct_residual(void *workspace, const Complex *boundary, const Complex *coupling,
                                            const Complex *solution, const Complex *normal, const Complex *rhs,
                                            double *relative) {
    auto &work = *static_cast<DirectWorkspace *>(workspace);
    CUDA_CHECK(cudaSetDevice(work.device));
    const int rc = direct_residual(work, boundary, coupling, solution, normal, rhs, relative);
    if (rc) cudaStreamSynchronize(work.stream);
    return rc;
}
