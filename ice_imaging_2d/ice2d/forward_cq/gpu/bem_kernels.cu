#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {

using Complex = cuDoubleComplex;
constexpr double pi = 3.14159265358979323846;
constexpr int threads = 128;

// Complex values use the same interleaved doubles as NumPy complex128. Store
// the real and imaginary parts separately here to keep the transfer record
// aligned to an eight-byte boundary independent of CUDA's double2 alignment.
struct NearSample {
    std::int32_t index;
    double argument_real, argument_imag;
    double single_real, single_imag;
    double double_real, double_imag;
};

struct Correction {
    std::int32_t index;
    double single_real, single_imag;
    double double_real, double_imag;
};

static_assert(sizeof(NearSample) == 56 && offsetof(NearSample, argument_real) == 8
              && offsetof(NearSample, single_real) == 24 && offsetof(NearSample, double_real) == 40);
static_assert(sizeof(Correction) == 40 && offsetof(Correction, single_real) == 8 && offsetof(Correction, double_real) == 24);

struct Workspace {
    int device;
    int panels;
    int double_columns;
    int tile_rows;
    int quadrature_count;
    int near_capacity = 0;
    Complex *geometry = nullptr;
    Complex *targets = nullptr;
    Complex *single = nullptr;
    Complex *dual = nullptr;
    double *weights = nullptr;
    int *near_count = nullptr;
    NearSample *near = nullptr;
    NearSample *near_host = nullptr;
    int *near_host_count = nullptr;
    int near_host_capacity = 0;
    int total_rows = 0;
    bool uploaded_targets = false;
    std::vector<int> previous_counts;
    int *correction_offsets = nullptr;
    int *correction_columns = nullptr;
    Correction *corrections = nullptr;
    Correction *correction_host = nullptr;
    int correction_capacity = 0;
    bool correction_pending = false;
    Complex scale_single = {};
    Complex scale_coupling = {};
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_end = nullptr;
    cudaStream_t stream = nullptr;

    ~Workspace() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        cudaFree(geometry);
        cudaFree(targets);
        cudaFree(single);
        cudaFree(dual);
        cudaFree(weights);
        cudaFree(near_count);
        cudaFree(near);
        if (near_host) cudaFreeHost(near_host);
        if (near_host_count) cudaFreeHost(near_host_count);
        cudaFree(correction_offsets);
        cudaFree(correction_columns);
        cudaFree(corrections);
        if (correction_host) cudaFreeHost(correction_host);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (kernel_end) cudaEventDestroy(kernel_end);
        if (stream) cudaStreamDestroy(stream);
    }
};

#define CUDA_CHECK(call) do { const auto status = (call); if (status != cudaSuccess) return status; } while (0)

__device__ Complex scaled(double factor, Complex value) {
    return make_cuDoubleComplex(factor * value.x, factor * value.y);
}

__device__ Complex square_root(Complex z) {
    const double absolute = hypot(z.x, z.y);
    if (z.x >= 0.0) {
        const double re = sqrt(0.5 * (absolute + z.x));
        return make_cuDoubleComplex(re, re == 0.0 ? z.y : 0.5 * z.y / re);
    }
    const double im = copysign(sqrt(0.5 * (absolute - z.x)), z.y);
    return make_cuDoubleComplex(0.5 * z.y / im, im);
}

template<int Order, int K>
__host__ __device__ constexpr double coefficient() {
    if constexpr (K == 0) {
        return 1.0;
    } else {
        constexpr double odd = 2.0 * K - 1.0;
        return coefficient<Order, K - 1>() * ((4.0 * Order * Order - odd * odd) / (8.0 * K));
    }
}

template<int Order, int K = 0>
__device__ __forceinline__ Complex polynomial(Complex inverse) {
    constexpr double constant = coefficient<Order, K>();
    if constexpr (K == 12) {
        return make_cuDoubleComplex(constant, 0.0);
    } else {
        const Complex higher = cuCmul(polynomial<Order, K + 1>(inverse), inverse);
        return make_cuDoubleComplex(higher.x + constant, higher.y);
    }
}

// Match the CPU kernel's K0/K1 expansion (DLMF 10.40.2). The host evaluates
// small and non-right-half-plane arguments with the same AMOS routine as CPU.
template<bool Resident>
__global__ void integrate(int first_row, int rows, int panels, int double_columns, int quadrature_count, Complex wave,
                          const Complex *xt, const Complex *zt, const Complex *geometry, const double *weights,
                          Complex *single, Complex *dual, int *near_count, NearSample *near, int near_capacity,
                          const int *correction_offsets, const int *correction_columns,
                          Complex scale_single, Complex scale_coupling) {
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= rows * panels) return;

    const int row = entry / panels;
    const int column = entry % panels;
    const int global_row = first_row + row;
    const int grid_points = panels * quadrature_count;
    const Complex *x = geometry;
    const Complex *z = x + grid_points;
    const Complex *length = z + grid_points;
    const Complex *nx = length + grid_points;
    const Complex *nz = nx + grid_points;
    Complex sum_single = make_cuDoubleComplex(0.0, 0.0);
    Complex sum_dual = sum_single;
    bool replaced = false;
    if (correction_offsets) {
        const int start = correction_offsets[global_row];
        const int stop = correction_offsets[global_row + 1];
        if (start < stop && column >= correction_columns[start] && column <= correction_columns[stop - 1]) {
            for (int k = start; k < stop; ++k) {
                if (column == correction_columns[k]) {
                    replaced = true;
                    break;
                }
            }
        }
    }
    for (int q = 0; q < quadrature_count && !replaced; ++q) {
        const int node = column * quadrature_count + q;
        const Complex dx = cuCsub(x[node], xt[row]);
        const Complex dz = cuCsub(z[node], zt[row]);
        const Complex radius = square_root(cuCadd(cuCmul(dx, dx), cuCmul(dz, dz)));
        const Complex argument = cuCmul(wave, radius);
        const Complex cs = scaled(weights[q], length[node]);
        Complex cd = make_cuDoubleComplex(0.0, 0.0);
        if (column < double_columns) {
            const Complex dot = cuCadd(cuCmul(dx, nx[node]), cuCmul(dz, nz[node]));
            if (dot.x != 0.0 || dot.y != 0.0) cd = cuCdiv(cuCmul(scaled(-weights[q], wave), dot), radius);
        }
        const double size_squared = argument.x * argument.x + argument.y * argument.y;
        if (argument.x < 0.0 || size_squared < 32.0 * 32.0) {
            const int index = atomicAdd(near_count, 1);
            if (index < near_capacity) near[index] = {entry, argument.x, argument.y, cs.x, cs.y, cd.x, cd.y};
            continue;
        }
        const Complex inverse = make_cuDoubleComplex(argument.x / size_squared, -argument.y / size_squared);
        double sine, cosine;
        sincos(-argument.y, &sine, &cosine);
        const Complex phase = make_cuDoubleComplex(cosine, sine);
        const Complex decay = scaled(exp(-argument.x), phase);
        const Complex prefactor = cuCmul(square_root(scaled(pi / 2.0, inverse)), decay);
        sum_single = cuCadd(sum_single, cuCmul(cuCmul(cs, prefactor), polynomial<0>(inverse)));
        if (cd.x != 0.0 || cd.y != 0.0) {
            sum_dual = cuCadd(sum_dual, cuCmul(cuCmul(cd, prefactor), polynomial<1>(inverse)));
        }
    }
    if constexpr (Resident) {
        if (column < double_columns) {
            if (global_row == column) sum_dual.x += 0.5;
            single[std::size_t(global_row) * panels + column] = sum_dual;
            dual[std::size_t(global_row) * double_columns + column] = cuCmul(scale_coupling, sum_single);
        } else {
            single[std::size_t(global_row) * panels + column] = cuCmul(scale_single, sum_single);
        }
    } else {
        single[entry] = sum_single;
        if (column < double_columns) dual[row * double_columns + column] = sum_dual;
    }
}

__device__ void add_complex(Complex *destination, Complex value) {
    atomicAdd(&destination->x, value.x);
    atomicAdd(&destination->y, value.y);
}

template<bool Replace>
__global__ void apply_corrections(int first_row, int count, int panels, int double_columns,
                                  const Correction *data, Complex *boundary, Complex *coupling,
                                  Complex scale_single, Complex scale_coupling) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= count) return;
    const auto record = data[sample];
    const int row = first_row + record.index / panels;
    const int column = record.index % panels;
    Complex *entry = boundary + std::size_t(row) * panels + column;
    const Complex single = make_cuDoubleComplex(record.single_real, record.single_imag);
    if (column < double_columns) {
        Complex *wet = coupling + std::size_t(row) * double_columns + column;
        const Complex scaled_single = cuCmul(scale_coupling, single);
        Complex dual = make_cuDoubleComplex(record.double_real, record.double_imag);
        if constexpr (Replace) {
            if (row == column) dual.x += 0.5;
            *entry = dual;
            *wet = scaled_single;
        } else {
            add_complex(entry, dual);
            add_complex(wet, scaled_single);
        }
    } else {
        const Complex scaled_single = cuCmul(scale_single, single);
        if constexpr (Replace) {
            *entry = scaled_single;
        } else {
            add_complex(entry, scaled_single);
        }
    }
}

int allocate_near(Workspace &work, int capacity) {
    CUDA_CHECK(cudaFree(work.near));
    work.near = nullptr;
    work.near_capacity = 0;
    CUDA_CHECK(cudaMalloc(&work.near, sizeof(NearSample) * static_cast<std::size_t>(capacity)));
    work.near_capacity = capacity;
    if (work.near_host_capacity < capacity) {
        if (work.near_host) CUDA_CHECK(cudaFreeHost(work.near_host));
        work.near_host = nullptr;
        work.near_host_capacity = 0;
        CUDA_CHECK(cudaMallocHost(&work.near_host, sizeof(NearSample) * static_cast<std::size_t>(capacity)));
        work.near_host_capacity = capacity;
    }
    return cudaSuccess;
}

}  // namespace

extern "C" int planar_cuda_create(int device, int panels, int double_columns, int tile_rows, int quadrature_count,
                                   const Complex *x, const Complex *z, const Complex *length,
                                   const Complex *nx, const Complex *nz, const double *weights, void **result) {
    *result = nullptr;
    if (panels < 1 || double_columns < 0 || double_columns > panels || tile_rows < 1 || quadrature_count < 1
        || static_cast<std::size_t>(panels) * tile_rows * quadrature_count > std::numeric_limits<int>::max()) {
        return cudaErrorInvalidValue;
    }
    CUDA_CHECK(cudaSetDevice(device));
    try {
        auto work = std::unique_ptr<Workspace>(new Workspace{device, panels, double_columns, tile_rows,
                                                             quadrature_count});
        CUDA_CHECK(cudaStreamCreateWithFlags(&work->stream, cudaStreamNonBlocking));
        const auto tile_entries = static_cast<std::size_t>(tile_rows) * panels;
        const auto points = static_cast<std::size_t>(panels) * quadrature_count;
        CUDA_CHECK(cudaMalloc(&work->geometry, 5 * points * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&work->targets, 2 * tile_rows * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&work->single, tile_entries * sizeof(Complex)));
        if (double_columns) CUDA_CHECK(cudaMalloc(&work->dual, std::size_t(tile_rows) * double_columns * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&work->weights, quadrature_count * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&work->near_count, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&work->near_host_count, sizeof(int)));
        const Complex *geometry[] = {x, z, length, nx, nz};
        for (int i = 0; i < 5; ++i) {
            CUDA_CHECK(cudaMemcpy(work->geometry + i * points, geometry[i], points * sizeof(Complex),
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(work->weights, weights, quadrature_count * sizeof(double), cudaMemcpyHostToDevice));
        const auto near_capacity = std::min(tile_entries * quadrature_count, std::max<std::size_t>(16384, tile_entries / 8));
        CUDA_CHECK(allocate_near(*work, static_cast<int>(near_capacity)));
        *result = work.release();
        return cudaSuccess;
    } catch (...) {
        return cudaErrorMemoryAllocation;
    }
}

extern "C" int planar_cuda_prepare(void *workspace, int rows, int corrections, const int *offsets, const int *columns,
                                   int resident, double single_real, double single_imag, double coupling_real,
                                   double coupling_imag, int profile, int near_capacity) {
    auto &work = *static_cast<Workspace *>(workspace);
    if (rows < 1 || corrections < 0 || near_capacity < 0
        || (resident && (rows != work.panels || work.double_columns == 0))) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaSetDevice(work.device));
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&work.kernel_start));
        CUDA_CHECK(cudaEventCreate(&work.kernel_end));
    }
    if (near_capacity > work.near_capacity) CUDA_CHECK(allocate_near(work, near_capacity));
    if (corrections) {
        CUDA_CHECK(cudaMalloc(&work.correction_offsets, (std::size_t(rows) + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&work.correction_columns, std::size_t(corrections) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(work.correction_offsets, offsets, (std::size_t(rows) + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(work.correction_columns, columns, std::size_t(corrections) * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (resident) {
        CUDA_CHECK(cudaFree(work.single));
        work.single = nullptr;
        CUDA_CHECK(cudaFree(work.dual));
        work.dual = nullptr;
        CUDA_CHECK(cudaMalloc(&work.single, std::size_t(rows) * work.panels * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&work.dual, std::size_t(rows) * work.double_columns * sizeof(Complex)));
        CUDA_CHECK(cudaFree(work.targets));
        work.targets = nullptr;
        CUDA_CHECK(cudaMalloc(&work.targets, 2 * std::size_t(rows) * sizeof(Complex)));
        try {
            work.previous_counts.resize(1 + (rows - 1) / work.tile_rows);
        } catch (...) {
            return cudaErrorMemoryAllocation;
        }
        work.total_rows = rows;
        work.scale_single = make_cuDoubleComplex(single_real, single_imag);
        work.scale_coupling = make_cuDoubleComplex(coupling_real, coupling_imag);
    }
    return cudaSuccess;
}

extern "C" void planar_cuda_destroy(void *workspace) {
    delete static_cast<Workspace *>(workspace);
}

extern "C" int planar_cuda_refresh(void *workspace, const Complex *x, const Complex *z, const Complex *length,
                                     const Complex *nx, const Complex *nz, const Complex *x_target,
                                     const Complex *z_target, double single_real, double single_imag,
                                     double coupling_real, double coupling_imag) {
    auto &work = *static_cast<Workspace *>(workspace);
    if (!work.total_rows) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaSetDevice(work.device));
    const auto points = std::size_t(work.panels) * work.quadrature_count;
    const Complex *geometry[] = {x, z, length, nx, nz};
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(work.geometry + i * points, geometry[i], points * sizeof(Complex),
                                   cudaMemcpyHostToDevice, work.stream));
    }
    // Reuse the device target storage across row tiles. Allocate its full
    // resident shape once, with the small staging buffer for host callers.
    const auto target_bytes = std::size_t(work.total_rows) * sizeof(Complex);
    CUDA_CHECK(cudaMemcpyAsync(work.targets, x_target, target_bytes, cudaMemcpyHostToDevice, work.stream));
    CUDA_CHECK(cudaMemcpyAsync(work.targets + work.total_rows, z_target, target_bytes, cudaMemcpyHostToDevice, work.stream));
    work.uploaded_targets = true;
    work.scale_single = make_cuDoubleComplex(single_real, single_imag);
    work.scale_coupling = make_cuDoubleComplex(coupling_real, coupling_imag);
    CUDA_CHECK(cudaStreamSynchronize(work.stream));
    work.correction_pending = false;
    return cudaSuccess;
}

static int assemble_rows(Workspace &work, int first_row, int rows, Complex wave,
                         const Complex *x_target, const Complex *z_target, Complex *single, Complex *dual,
                         const void **near_field, int *near_size, double *kernel_seconds) {
    if (rows < 1 || rows > work.tile_rows || first_row < 0
        || (work.total_rows && first_row + rows > work.total_rows)) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaSetDevice(work.device));
    const Complex *xt = work.targets;
    const Complex *zt = work.targets + rows;
    if (work.uploaded_targets) {
        xt += first_row;
        zt = work.targets + work.total_rows + first_row;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(work.targets, x_target, rows * sizeof(Complex), cudaMemcpyHostToDevice, work.stream));
        CUDA_CHECK(cudaMemcpyAsync(work.targets + rows, z_target, rows * sizeof(Complex), cudaMemcpyHostToDevice, work.stream));
    }
    const int entries = rows * work.panels;
    int count = 0;
    int downloaded = 0;
    for (;;) {
        CUDA_CHECK(cudaMemsetAsync(work.near_count, 0, sizeof(int), work.stream));
        if (work.kernel_start) CUDA_CHECK(cudaEventRecord(work.kernel_start, work.stream));
        const int blocks = 1 + (entries - 1) / threads;
        if (work.total_rows) {
            integrate<true><<<blocks, threads, 0, work.stream>>>(first_row, rows, work.panels, work.double_columns,
                work.quadrature_count, wave, xt, zt, work.geometry, work.weights, work.single,
                work.dual, work.near_count, work.near, work.near_capacity, work.correction_offsets,
                work.correction_columns, work.scale_single, work.scale_coupling);
        } else {
            integrate<false><<<blocks, threads, 0, work.stream>>>(first_row, rows, work.panels, work.double_columns,
                work.quadrature_count, wave, xt, zt, work.geometry, work.weights, work.single,
                work.dual, work.near_count, work.near, work.near_capacity, work.correction_offsets,
                work.correction_columns, work.scale_single, work.scale_coupling);
        }
        CUDA_CHECK(cudaGetLastError());
        if (work.kernel_end) CUDA_CHECK(cudaEventRecord(work.kernel_end, work.stream));
        CUDA_CHECK(cudaMemcpyAsync(work.near_host_count, work.near_count, sizeof(int), cudaMemcpyDeviceToHost, work.stream));
        downloaded = 0;
        if (work.uploaded_targets) {
            const int previous = work.previous_counts[first_row / work.tile_rows];
            // A fixed prefetch bound changes only transfer size. If it is too
            // small, fetch the remainder after reading the actual count.
            const auto estimate = std::int64_t(previous) + previous / 8 + 4096;
            downloaded = static_cast<int>(std::min<std::int64_t>(work.near_capacity, estimate));
            if (downloaded) CUDA_CHECK(cudaMemcpyAsync(work.near_host, work.near, sizeof(NearSample) * downloaded,
                                                      cudaMemcpyDeviceToHost, work.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(work.stream));
        work.correction_pending = false;
        count = *work.near_host_count;
        if (work.kernel_end) {
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, work.kernel_start, work.kernel_end));
            *kernel_seconds += milliseconds * 0.001;
        }
        if (count <= work.near_capacity) break;
        CUDA_CHECK(allocate_near(work, count));
        if (work.uploaded_targets) work.previous_counts[first_row / work.tile_rows] = count;
    }

    if (work.uploaded_targets) work.previous_counts[first_row / work.tile_rows] = count;

    if (!work.total_rows) {
        CUDA_CHECK(cudaMemcpy(single, work.single, sizeof(Complex) * entries, cudaMemcpyDeviceToHost));
        if (work.double_columns) {
            CUDA_CHECK(cudaMemcpy(dual, work.dual, sizeof(Complex) * std::size_t(rows) * work.double_columns,
                                  cudaMemcpyDeviceToHost));
        }
    }
    if (count > downloaded) {
        CUDA_CHECK(cudaMemcpyAsync(work.near_host + downloaded, work.near + downloaded,
                                   sizeof(NearSample) * (count - downloaded), cudaMemcpyDeviceToHost, work.stream));
        CUDA_CHECK(cudaStreamSynchronize(work.stream));
    }
    *near_field = work.near_host;
    *near_size = count;
    return cudaSuccess;
}

extern "C" int planar_cuda_rows(void *workspace, int rows, double wave_real, double wave_imag,
                                 const Complex *x_target, const Complex *z_target, Complex *single, Complex *dual,
                                 const void **near_field, int *near_size) {
    double seconds = 0;
    return assemble_rows(*static_cast<Workspace *>(workspace), 0, rows, make_cuDoubleComplex(wave_real, wave_imag),
                         x_target, z_target, single, dual, near_field, near_size, &seconds);
}

extern "C" int planar_cuda_rows_v2(void *workspace, int first_row, int rows, double wave_real, double wave_imag,
                                    const Complex *x_target, const Complex *z_target, Complex *single, Complex *dual,
                                    const void **near_field, int *near_size, double *kernel_seconds) {
    *kernel_seconds = 0;
    return assemble_rows(*static_cast<Workspace *>(workspace), first_row, rows, make_cuDoubleComplex(wave_real, wave_imag),
                         x_target, z_target, single, dual, near_field, near_size, kernel_seconds);
}

extern "C" int planar_cuda_correct(void *workspace, int first_row, int count, const void *data, int replace) {
    auto &work = *static_cast<Workspace *>(workspace);
    if (!work.total_rows || count < 0 || first_row < 0 || first_row >= work.total_rows) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    CUDA_CHECK(cudaSetDevice(work.device));
    // Every native row call synchronizes this stream before returning its
    // AMOS payload. For consecutive corrections, synchronize before reusing
    // the pinned upload buffer (the upload+scatter itself can run asynchronously).
    if (work.correction_pending) CUDA_CHECK(cudaStreamSynchronize(work.stream));
    if (count > work.correction_capacity) {
        CUDA_CHECK(cudaFree(work.corrections));
        work.corrections = nullptr;
        if (work.correction_host) CUDA_CHECK(cudaFreeHost(work.correction_host));
        work.correction_host = nullptr;
        work.correction_capacity = 0;
        CUDA_CHECK(cudaMalloc(&work.corrections, std::size_t(count) * sizeof(Correction)));
        CUDA_CHECK(cudaMallocHost(&work.correction_host, std::size_t(count) * sizeof(Correction)));
        work.correction_capacity = count;
    }
    const auto bytes = std::size_t(count) * sizeof(Correction);
    std::memcpy(work.correction_host, data, bytes);
    CUDA_CHECK(cudaMemcpyAsync(work.corrections, work.correction_host, bytes,
                               cudaMemcpyHostToDevice, work.stream));
    work.correction_pending = true;
    const int blocks = 1 + (count - 1) / threads;
    if (replace) {
        apply_corrections<true><<<blocks, threads, 0, work.stream>>>(first_row, count, work.panels, work.double_columns, work.corrections,
             work.single, work.dual, work.scale_single, work.scale_coupling);
    } else {
        apply_corrections<false><<<blocks, threads, 0, work.stream>>>(first_row, count, work.panels, work.double_columns, work.corrections,
             work.single, work.dual, work.scale_single, work.scale_coupling);
    }
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

extern "C" int planar_cuda_boundary_pointers(void *workspace, void **boundary, void **coupling) {
    auto &work = *static_cast<Workspace *>(workspace);
    if (!work.total_rows) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaSetDevice(work.device));
    CUDA_CHECK(cudaStreamSynchronize(work.stream));
    work.correction_pending = false;
    *boundary = work.single;
    *coupling = work.dual;
    return cudaSuccess;
}

extern "C" const char *planar_cuda_error(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
