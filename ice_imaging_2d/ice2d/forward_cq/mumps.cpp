#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <zmumps_c.h>

namespace {
static_assert(sizeof(ZMUMPS_COMPLEX) == 2 * sizeof(double));
using Mumps = ZMUMPS_STRUC_C;

int report(const Mumps &solver, int64_t *out) {
    constexpr int fields[] = {1, 2, 7, 16, 17, 20, 21, 22, 26, 29};
    for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
        out[i] = solver.infog[fields[i] - 1];
    return solver.infog[0];
}

template<typename Index>
int64_t extract_upper(int32_t n, const Index *offset, const Index *rows, const ZMUMPS_COMPLEX *values,
                      int64_t capacity, MUMPS_INT *out_rows, MUMPS_INT *out_cols,
                      ZMUMPS_COMPLEX *out_values, bool reuse_pattern) {
    int64_t k = 0;
    for (int32_t j = 0; j < n; ++j) {
        const Index *begin = rows + offset[j];
        const Index *end = std::upper_bound(begin, rows + offset[j + 1], j);
        if (!out_values) {
            k += end - begin;
            continue;
        }
        if (end - begin > capacity - k) return -1;
        for (const Index *row = begin; row != end; ++row, ++k) {
            // Present the lower triangle by transposing the complex symmetric input.
            if (reuse_pattern) {
                if (out_rows[k] != j + 1 || out_cols[k] != *row + 1) return -1;
            } else {
                out_rows[k] = j + 1;
                out_cols[k] = *row + 1;
            }
            out_values[k] = values[row - rows];
        }
    }
    return k;
}
} // namespace

extern "C" int planar_mumps_index_bytes() {
    return sizeof(MUMPS_INT);
}

extern "C" int64_t planar_mumps_upper(int32_t n, int index_bytes, const void *offset,
                                      const void *rows, const void *values, int64_t capacity,
                                      MUMPS_INT *out_rows, MUMPS_INT *out_cols, void *out_values,
                                      int reuse_pattern) {
    const auto *a = static_cast<const ZMUMPS_COMPLEX *>(values);
    auto *b = static_cast<ZMUMPS_COMPLEX *>(out_values);
    if (index_bytes == 4)
        return extract_upper(n, static_cast<const int32_t *>(offset), static_cast<const int32_t *>(rows),
                             a, capacity, out_rows, out_cols, b, reuse_pattern);
    if (index_bytes == 8)
        return extract_upper(n, static_cast<const int64_t *>(offset), static_cast<const int64_t *>(rows),
                             a, capacity, out_rows, out_cols, b, reuse_pattern);
    return -2;
}

extern "C" void *planar_mumps_create(int32_t n, int64_t nnz, MUMPS_INT *rows, MUMPS_INT *columns,
                                      int ordering, int memory_mb, const char *out_of_core) {
    auto solver = std::unique_ptr<Mumps>(new (std::nothrow) Mumps{});
    if (!solver) return nullptr;
    solver->sym = 2; // complex symmetric, not Hermitian
    solver->par = 1;
    solver->comm_fortran = -987654; // MPI_COMM_WORLD using the sequential MPI library
    solver->job = -1;
    zmumps_c(solver.get());
    if (solver->infog[0] < 0) return nullptr;
    solver->icntl[0] = -1;
    solver->icntl[1] = -1;
    solver->icntl[2] = -1;
    solver->icntl[3] = 0;
    solver->icntl[5] = 0; // no value-dependent permutation: reuse analysis between s samples
    solver->icntl[6] = ordering;
    solver->icntl[11] = 1; // do not activate strongly indefinite value-dependent matching
    solver->icntl[22] = memory_mb;
    solver->n = n;
    solver->nnz = nnz;
    solver->irn = rows;
    solver->jcn = columns;
    if (out_of_core && out_of_core[0]) {
        solver->icntl[21] = 1;
        std::snprintf(solver->ooc_tmpdir, sizeof(solver->ooc_tmpdir), "%s", out_of_core);
        std::snprintf(solver->ooc_prefix, sizeof(solver->ooc_prefix), "planar_ice");
    }
    return solver.release();
}

extern "C" int planar_mumps_phase(void *handle, int job, void *values, int64_t *out) {
    auto &solver = *static_cast<Mumps *>(handle);
    solver.a = static_cast<ZMUMPS_COMPLEX *>(values);
    solver.job = job;
    zmumps_c(&solver);
    return report(solver, out);
}

extern "C" int planar_mumps_solve(void *handle, void *rhs, int32_t rhs_count, int64_t *out) {
    auto &solver = *static_cast<Mumps *>(handle);
    solver.rhs = static_cast<ZMUMPS_COMPLEX *>(rhs);
    solver.nrhs = rhs_count;
    solver.lrhs = solver.n;
    solver.job = 3;
    zmumps_c(&solver);
    solver.rhs = nullptr;
    return report(solver, out);
}

extern "C" int planar_mumps_solve_sparse(void *handle, MUMPS_INT *indices, void *values,
                                         int32_t nonzero, void *dense_solution, int64_t *out) {
    auto &solver = *static_cast<Mumps *>(handle);
    const auto old_format = solver.icntl[19];
    MUMPS_INT offsets[] = {1, static_cast<MUMPS_INT>(nonzero) + 1};
    solver.icntl[19] = 1;
    solver.irhs_ptr = offsets;
    solver.irhs_sparse = indices;
    solver.rhs_sparse = static_cast<ZMUMPS_COMPLEX *>(values);
    solver.nz_rhs = nonzero;
    solver.rhs = static_cast<ZMUMPS_COMPLEX *>(dense_solution);
    solver.nrhs = 1;
    solver.lrhs = solver.n;
    solver.job = 3;
    zmumps_c(&solver);
    solver.icntl[19] = old_format;
    solver.rhs = solver.rhs_sparse = nullptr;
    solver.irhs_ptr = solver.irhs_sparse = nullptr;
    solver.nz_rhs = 0;
    return report(solver, out);
}

extern "C" void planar_mumps_destroy(void *handle) {
    std::unique_ptr<Mumps> solver(static_cast<Mumps *>(handle));
    if (!solver) return;
    solver->job = -2;
    zmumps_c(solver.get());
}
