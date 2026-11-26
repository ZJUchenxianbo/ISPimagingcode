#include <cstddef>
#include <cstdint>

extern "C" void planar_ice_physical(std::size_t count, const double *stiffness, const double *mass,
                                    double wave_re, double wave_im, double *values) {
    for (std::size_t i = 0; i < count; ++i) {
        const double real_mass = mass[i];
        values[2 * i] = real_mass != 0.0 ? stiffness[i] + wave_re * real_mass : stiffness[i];
        values[2 * i + 1] = real_mass != 0.0 ? 0.0 + wave_im * real_mass : 0.0;
    }
}

extern "C" void planar_ice_pml(std::size_t count, const std::int32_t *sources, const std::int32_t *targets,
                               const double *constant, const double *linear, const double *quadratic,
                               double fraction, double *values) {
    // Preserve summation order when two elements contribute to the same slot.
    if (!linear || fraction == 0.0) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto source = std::size_t(sources[i]) * 2;
            const auto target = std::size_t(targets[i]) * 2;
            values[target] += constant[source];
            values[target + 1] += constant[source + 1];
        }
        return;
    }

    const double shifted = fraction - 1.0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto source = std::size_t(sources[i]) * 2;
        const auto target = std::size_t(targets[i]) * 2;
        const double real = constant[source] + fraction * (linear[source] + shifted * quadratic[source]);
        const double imag = constant[source + 1] + fraction * (linear[source + 1] + shifted * quadratic[source + 1]);
        values[target] += real;
        values[target + 1] += imag;
    }
}
