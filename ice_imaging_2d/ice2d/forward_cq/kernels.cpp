#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace {
using Complex = std::complex<double>;

constexpr double pi = 3.14159265358979323846;
constexpr int terms = 13;

constexpr std::array<double, terms> coefficients(double order) {
    std::array<double, terms> result{};
    result[0] = 1.0;
    for (int k = 1; k < terms; ++k) {
        const double odd = 2.0 * k - 1.0;
        result[k] = result[k - 1] * ((4.0 * order * order - odd * odd) / (8.0 * k));
    }
    return result;
}

constexpr auto k0_coefficients = coefficients(0.0);
constexpr auto k1_coefficients = coefficients(1.0);

Complex polynomial(Complex inv_z, const std::array<double, terms> &coefficients) {
    Complex value = coefficients.back();
    for (int k = terms - 2; k >= 0; --k) {
        value = value * inv_z + coefficients[k];
    }
    return value;
}

struct NearField {
    std::vector<std::int32_t> indices;
    std::vector<Complex> arguments;
    std::vector<Complex> single_coefficients;
    std::vector<Complex> double_coefficients;
};
}

// AMOS handles the compact list of small or non-right-half-plane arguments.
// NIST DLMF 10.40.2 gives the large-argument expansions on the right half-plane.
extern "C" void *planar_accumulate_regular(int n_targets, int n_panels, int quadrature_count, int n_double,
                                         double wave_real, double wave_imag,
                                         const Complex *xt, const Complex *zt, const Complex *x, const Complex *z,
                                         const Complex *length, const Complex *nx, const Complex *nz,
                                         const double *weights, Complex *single, Complex *dual) {
    try {
        auto near = std::make_unique<NearField>();
        const Complex wave(wave_real, wave_imag);
        for (int row = 0; row < n_targets; ++row) {
            for (int col = 0; col < n_panels; ++col) {
                Complex sum_s(0.0), sum_d(0.0);
                for (int q = 0; q < quadrature_count; ++q) {
                    const int pos = col * quadrature_count + q;
                    const Complex dx = x[pos] - xt[row];
                    const Complex dz = z[pos] - zt[row];
                    const Complex radius = std::sqrt(dx * dx + dz * dz);
                    const Complex arg = wave * radius;
                    const Complex cs = weights[q] * length[pos];
                    Complex cd(0.0);
                    if (col < n_double) {
                        const Complex dot = dx * nx[pos] + dz * nz[pos];
                        if (dot != Complex(0.0)) cd = -weights[q] * wave * dot / radius;
                    }
                    const double scale = std::norm(arg);
                    if (arg.real() < 0.0 || scale < 32.0 * 32.0) {
                        near->indices.push_back(row * n_panels + col);
                        near->arguments.push_back(arg);
                        near->single_coefficients.push_back(cs);
                        near->double_coefficients.push_back(cd);
                    } else {
                        const Complex inv_z = Complex(arg.real(), -arg.imag()) / scale;
                        const Complex prefactor = std::sqrt((pi / 2.0) * inv_z) * std::exp(-arg);
                        sum_s += cs * prefactor * polynomial(inv_z, k0_coefficients);
                        if (cd != Complex(0.0)) sum_d += cd * prefactor * polynomial(inv_z, k1_coefficients);
                    }
                }
                single[row * n_panels + col] = sum_s;
                if (col < n_double) dual[row * n_double + col] = sum_d;
            }
        }
        return near.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" std::size_t planar_near_fields(const void *ptr, const std::int32_t **indices,
                                          const Complex **arguments, const Complex **single_coefficients,
                                          const Complex **double_coefficients) {
    const auto *near = static_cast<const NearField *>(ptr);
    *indices = near->indices.data();
    *arguments = near->arguments.data();
    *single_coefficients = near->single_coefficients.data();
    *double_coefficients = near->double_coefficients.data();
    return near->indices.size();
}

extern "C" void planar_free_near_fields(void *ptr) {
    delete static_cast<NearField *>(ptr);
}
