#include <cstddef>

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace {

__global__ void assemble_physical(int count, const double *stiffness, const double *mass,
                                  double wave_re, double wave_im, cuDoubleComplex *values) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count)) return;
    const double m = mass[i];
    values[i] = m != 0.0 ? make_cuDoubleComplex(stiffness[i] + wave_re * m, 0.0 + wave_im * m)
                         : make_cuDoubleComplex(stiffness[i], 0.0);
}

__global__ void assemble_pml(int count, const int *targets, const int *offsets, const int *sources, const int *rows,
                             int degree, int coefficients, const cuDoubleComplex *constant, cuDoubleComplex *values) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count)) return;
    const auto *linear = constant + coefficients;
    const auto *quadratic = linear + coefficients;
    const int dest = targets[i];
    auto result = values[dest];
    // Each target has a single owner. Traverse contributions in the host assembly order.
    for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
        const int source = sources[k], row = rows[k];
        auto entry = constant[source];
        if (degree && row) {
            const double z = double(row) / degree, shift = z - 1.0;
            entry.x += z * (linear[source].x + shift * quadratic[source].x);
            entry.y += z * (linear[source].y + shift * quadratic[source].y);
        }
        result.x += entry.x;
        result.y += entry.y;
    }
    values[dest] = result;
}

}  // namespace

extern "C" cudaError_t launch_panel_assembly(cudaStream_t stream, int count, const double *stiffness,
                                              const double *mass, double wave_re, double wave_im,
                                              int targets_count, const int *targets, const int *offsets,
                                              const int *sources, const int *rows, int degree,
                                              int coefficients, const cuDoubleComplex *pml, cuDoubleComplex *values) {
    constexpr int tile = 256;
    assemble_physical<<<(count - 1) / tile + 1, tile, 0, stream>>>(count, stiffness, mass, wave_re, wave_im, values);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    if (targets_count) {
        assemble_pml<<<(targets_count - 1) / tile + 1, tile, 0, stream>>>(targets_count, targets, offsets, sources,
                                                                       rows, degree, coefficients, pml, values);
        return cudaGetLastError();
    }
    return cudaSuccess;
}
