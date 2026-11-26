"""Eliminate elastic displacement directly onto the acoustic pressure panels."""

import ctypes
from functools import lru_cache
from operator import index
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np


DEFAULT_LIBRARY = Path(__file__).with_name("libgpu_panel_schur.so")
NATIVE_PHASES = ("ice_panel_upload_s", "ice_panel_numeric_s", "ice_panel_extract_s", "ice_panel_download_s")


def build_library(cudss_root, *, cuda_root=None, output=DEFAULT_LIBRARY,
                  device_assembly=False, cuda_arch=None):
    cuda_root = cuda_root or os.environ.get('CUDA_ROOT', '/usr/local/cuda')
    cuda_arch = cuda_arch or os.environ.get('PLANAR_CUDA_ARCH')
    if device_assembly and not cuda_arch:
        raise ValueError('Specify cuda_arch or PLANAR_CUDA_ARCH for the target GPU')
    cuda_root, cudss_root = Path(cuda_root), Path(cudss_root)
    source = Path(__file__).with_suffix(".cpp")
    if device_assembly:
        compile_flags = [str(cuda_root / "bin/nvcc"), "-std=c++17", "-O3", "--fmad=false",
                         f"-arch={cuda_arch}", "-shared", "-Xcompiler", "-fPIC", "-DPANEL_DEVICE_ASSEMBLY",
                         str(source.with_name("panel_assembly.cu"))]
    else:
        compile_flags = ["c++", "-std=c++17", "-shared", "-fPIC", "-O3"]
    subprocess.run([
        *compile_flags, str(source),
        f"-I{cuda_root / 'include'}", f"-I{cudss_root / 'include'}",
        f"-L{cuda_root / 'targets/x86_64-linux/lib'}", f"-L{cudss_root / 'lib'}",
        "-lcudart", "-lcudss", "-lcusparse", "-lcublas", "-o", str(output),
    ], check=True)
    return Path(output).resolve()


@lru_cache(maxsize=2)
def _library(path):
    lib = ctypes.CDLL(str(path))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.gpu_panel_create.argtypes = (integer, integer, integer, integer, integer,
                                     ptr, ptr, ptr, ctypes.POINTER(ptr))
    lib.gpu_panel_create.restype = integer
    if hasattr(lib, "gpu_panel_create_configured"):
        lib.gpu_panel_create_configured.argtypes = (integer, integer, integer, integer, integer, ptr, ptr, ptr,
                                                     integer, integer, integer, ptr, ctypes.POINTER(ptr))
        lib.gpu_panel_create_configured.restype = integer
    lib.gpu_panel_destroy.argtypes = (ptr,)
    lib.gpu_panel_destroy.restype = None
    lib.gpu_panel_factor.argtypes = (ptr, ptr, ptr, ptr, ptr)
    lib.gpu_panel_factor.restype = integer
    lib.gpu_panel_solve.argtypes = (ptr, ptr, ptr)
    lib.gpu_panel_solve.restype = integer
    if hasattr(lib, "gpu_panel_solve_normal"):
        lib.gpu_panel_solve_normal.argtypes = (ptr, ptr, ptr, ptr)
        lib.gpu_panel_solve_normal.restype = integer
    for name in ("gpu_panel_solve_normal_many", "gpu_panel_solve_normal_many_device"):
        function = getattr(lib, name, None)
        if function is not None:
            function.argtypes = (ptr, integer, ptr, ptr, ptr, ptr)
            function.restype = integer
    if hasattr(lib, "gpu_panel_original_residual"):
        lib.gpu_panel_original_residual.argtypes = (ptr, ptr, ptr, ptr)
        lib.gpu_panel_original_residual.restype = integer
    if hasattr(lib, "gpu_panel_assembly_setup"):
        lib.gpu_panel_assembly_setup.argtypes = (ptr, ptr, ptr, integer, ptr, ptr, integer, ptr, ptr,
                                                 integer, integer, integer)
        lib.gpu_panel_assembly_setup.restype = integer
        lib.gpu_panel_factor_assembled.argtypes = (ptr, ptr, ctypes.c_double, ctypes.c_double, ptr, ptr, ptr, ptr, ptr)
        lib.gpu_panel_factor_assembled.restype = integer
        lib.gpu_panel_read_elastic_values.argtypes = (ptr, ptr)
        lib.gpu_panel_read_elastic_values.restype = integer
    return lib


def _same(a, b):
    return a is b or np.array_equal(a, b)


def _check(code, stage):
    if code:
        raise RuntimeError(f"CUDA cuDSS panel Schur {stage} failed (native status {code})")


class GpuPanelSchur:
    """Factor [0, L.T; L, A] with cuDSS retaining only the panel unknowns.

    Here A is the original elastic matrix and L maps pressure to the elastic
    load. cuDSS eliminates A, leaving the exact panel system -L.T @ inv(A) @ L.
    The panel rows come first so the elastic values remain a contiguous block.
    """

    def __init__(self, library_path=DEFAULT_LIBRARY, *, device=0, nd_levels=0, amd=False, superpanels=True):
        self.device = int(device)
        self._workspace = ctypes.c_void_p()
        self._signature = None
        self._diagonal_positions = None
        self._assembly_layout = None
        self._generation = 0
        self._lib = _library(Path(library_path).resolve())
        self._reordering = (int(amd), int(nd_levels), int(superpanels))
        if nd_levels < 0 or (amd and nd_levels):
            raise ValueError("Choose either AMD or a positive nested-dissection depth")
        self.analysis_stats = {}

    def close(self):
        if self._workspace:
            self._lib.gpu_panel_destroy(self._workspace)
        self._workspace = ctypes.c_void_p()
        self._signature = None
        self._diagonal_positions = None
        self._assembly_layout = None

    def __del__(self):
        self.close()

    def _create(self, n, panels, offsets, columns, trace):
        limit = np.iinfo(np.int32).max
        head = trace.nnz + panels
        if max(n + panels, head + columns.size) >= limit:
            raise ValueError("The augmented CUDA matrix exceeds 32-bit CSR limits")
        diagonal_positions = trace.indptr[:-1] + np.arange(panels, dtype=np.int32)
        coupling_positions = np.ones(head, dtype=bool)
        coupling_positions[diagonal_positions] = False
        index = np.empty(head + columns.size, dtype=np.int32)
        index[diagonal_positions] = np.arange(panels, dtype=np.int32)
        index[:head][coupling_positions] = trace.indices + panels
        index[head:] = columns + panels
        row = np.empty(n + panels + 1, dtype=np.int32)
        row[:panels] = diagonal_positions
        row[panels:] = offsets + head
        marks = np.zeros(n + panels, dtype=np.int32)
        marks[:panels] = 1
        args = (self.device, n, panels, head, columns.size, row.ctypes.data, index.ctypes.data, marks.ctypes.data)
        out = ctypes.byref(self._workspace)
        if hasattr(self._lib, "gpu_panel_create_configured"):
            stats = np.full(2, -1, dtype=np.int64)
            _check(self._lib.gpu_panel_create_configured(*args, *self._reordering, stats.ctypes.data, out), "analysis")
            self.analysis_stats = dict(zip(("estimated_device_peak_bytes", "factor_nnz"), map(int, stats)))
        else:
            if self._reordering != (0, 0, 1):
                raise RuntimeError("Rebuild the panel Schur CUDA library to change reordering")
            _check(self._lib.gpu_panel_create(*args, out), "analysis")
        self._coupling_positions = coupling_positions

    def _diagonal(self):
        if self._diagonal_positions is None:
            offsets, column, _, _ = self._signature
            starts = offsets[:-1]
            if np.any(np.diff(offsets) <= 0) or np.any(column[starts] != np.arange(starts.size)):
                raise ValueError("GPU residual requires every elastic diagonal in the upper CSR")
            self._diagonal_positions = (starts + self._coupling_positions.size).astype(np.int32, copy=False)
        return self._diagonal_positions

    def factor(self, s, load, average, *, matrix, upper_csr, timings=None):
        """Export the exact displacement response to one pressure per panel."""
        offsets, columns, values = upper_csr
        if values.dtype != np.complex128:
            raise ValueError("Panel Schur requires complex128 values")
        if matrix.shape != (load.shape[0], load.shape[0]):
            raise ValueError("Incompatible ice matrix shape")
        return self._factor(s, load, average, matrix, offsets, columns, values, None, timings)

    def factor_assembled(self, s, load, average, *, assembler, timings=None):
        """Build and factor the original elastic CSR directly on the GPU."""
        if not hasattr(self._lib, "gpu_panel_factor_assembled"):
            raise RuntimeError("Rebuild the panel Schur CUDA library with device_assembly=True")
        offsets, columns = assembler.upper_pattern
        if assembler.ndof != load.shape[0]:
            raise ValueError("The ice assembler and panel load have different dimensions")
        return self._factor(s, load, average, None, offsets, columns, None, assembler, timings)

    def _factor(self, s, load, average, matrix, offsets, columns, values, assembler, timings):
        self._generation += 1
        n, panels = load.shape
        if n + 1 != offsets.size or average.shape != (panels, n):
            raise ValueError("Incompatible ice and panel operators")
        if offsets.dtype != np.int32 or columns.dtype != np.int32:
            raise ValueError("Panel Schur requires 32-bit CSR indices")
        trace = load.T.tocsr()
        trace.sum_duplicates()
        trace.sort_indices()
        average = average.tocsr()
        average.sort_indices()
        if not (_same(trace.indptr, average.indptr) and _same(trace.indices, average.indices)):
            raise ValueError("Panel Schur requires matching average and transpose load support")
        if np.any(np.diff(trace.indptr) < 1):
            raise ValueError("Every acoustic panel must load at least one elastic DOF")
        starts = trace.indptr[:-1]
        scale = average.data[starts] / trace.data[starts]
        ratios = np.repeat(scale, np.diff(trace.indptr))
        if not np.allclose(average.data, ratios * trace.data, rtol=5e-14, atol=0):
            raise ValueError("Each panel average must be proportional to the transpose load")

        key = offsets, columns, trace.indptr, trace.indices
        reused = self._signature is not None and all(_same(a, b) for a, b in zip(self._signature, key))
        if timings is not None:
            timings["ice_gpu_symbolic_reused"] = int(reused)
        if not reused:
            began = perf_counter()
            self.close()
            self._create(n, panels, offsets, columns, trace)
            self._signature = key
            if timings is not None:
                timings["ice_gpu_symbolic_s"] = perf_counter() - began

        head = np.zeros(self._coupling_positions.size, dtype=np.complex128)
        head[self._coupling_positions] = trace.data
        schur = np.empty((panels, panels), dtype=np.complex128, order="F")
        native = np.empty(len(NATIVE_PHASES), dtype=np.float64) if timings is not None else None
        times_ptr = native.ctypes.data if native is not None else None
        began = perf_counter()
        if assembler is None:
            code = self._lib.gpu_panel_factor(self._workspace, head.ctypes.data, values.ctypes.data,
                                              schur.ctypes.data, times_ptr)
        else:
            if self._assembly_layout is not assembler:
                self._register_assembly(assembler)
                if timings is not None:
                    timings["ice_panel_device_setup_s"] = perf_counter() - began
            started = perf_counter()
            coefficients = assembler.pml_coefficients(s)
            pointers = [values.ctypes.data if values is not None else None for values in coefficients]
            if timings is not None:
                timings["ice_panel_pml_coefficients_s"] = perf_counter() - started
            began = perf_counter()
            wave = s**2
            code = self._lib.gpu_panel_factor_assembled(self._workspace, head.ctypes.data, wave.real, wave.imag,
                                                       *pointers, schur.ctypes.data, times_ptr)
        if code or not np.isfinite(schur).all():
            self.close()
            _check(code, "numeric factorization")
            raise RuntimeError("cuDSS exported a nonfinite panel Schur complement")
        response = np.ascontiguousarray(-scale[:, None] * schur)
        if timings is not None:
            timings.update(zip(NATIVE_PHASES, native))
            timings["ice_gpu_numeric_and_response_s"] = perf_counter() - began
        return _NumericPanelFactor(self, self._generation, n, matrix, response, scale)

    def _register_assembly(self, assembler):
        stiff, mass, targets, offsets, sources, rows, degree, count, levels = assembler.prepare_device_assembly()
        _check(self._lib.gpu_panel_assembly_setup(self._workspace, stiff.ctypes.data, mass.ctypes.data, len(targets),
                                                 targets.ctypes.data, offsets.ctypes.data, len(sources),
                                                 sources.ctypes.data, rows.ctypes.data, degree, count, levels),
               "device assembly setup")
        self._assembly_layout = assembler


class _NumericPanelFactor:
    def __init__(self, owner, generation, size, matrix, response, scale):
        self.owner = owner
        self.generation = generation
        self.size = size
        self.matrix = matrix
        self.response = response
        self._scale = scale
        self._rhs = None
        self._panel_solved = False

    def panel_response(self):
        return self.response

    def assembled_values(self):
        """Copy the GPU's assembled upper triangle for comparison with a host reference."""
        owner = self.owner
        if not owner._workspace or self.generation != owner._generation or owner._assembly_layout is None:
            raise RuntimeError("No matching device-assembled ice matrix")
        values = np.empty(owner._signature[1].size, dtype=np.complex128)
        _check(owner._lib.gpu_panel_read_elastic_values(owner._workspace, values.ctypes.data), "read device assembly")
        return values

    def solve(self, rhs):
        """Recover full elastic displacement for the independently checked RHS."""
        owner = self.owner
        if not owner._workspace or self.generation != owner._generation:
            raise RuntimeError("CUDA panel Schur factor is stale")
        rhs = np.ascontiguousarray(rhs, dtype=np.complex128)
        if rhs.shape != (self.size,):
            raise ValueError("Expected one original ice right-hand side")
        solution = np.empty_like(rhs)
        code = owner._lib.gpu_panel_solve(owner._workspace, rhs.ctypes.data, solution.ctypes.data)
        if code or not np.isfinite(solution).all():
            owner.close()
            _check(code, "elastic recovery")
            raise RuntimeError("cuDSS recovered a nonfinite elastic displacement")
        self._rhs = rhs.copy()
        self._panel_solved = False
        return solution

    def solve_panel_normal(self, pressure):
        """Apply the physical face load and recover only the normal panel displacement on CUDA."""
        owner = self.owner
        if not owner._workspace or self.generation != owner._generation:
            raise RuntimeError("CUDA panel Schur factor is stale")
        if not hasattr(owner._lib, "gpu_panel_solve_normal"):
            raise RuntimeError("Rebuild the panel Schur CUDA library for GPU panel recovery")
        pressure = np.ascontiguousarray(pressure, dtype=np.complex128)
        if pressure.shape != self._scale.shape:
            raise ValueError("Expected one pressure per acoustic panel")
        normal = np.empty_like(pressure)
        code = owner._lib.gpu_panel_solve_normal(owner._workspace, pressure.ctypes.data,
                                               owner._diagonal().ctypes.data, normal.ctypes.data)
        if code:
            owner.close()
            _check(code, "panel recovery")
        normal *= self._scale
        if not np.isfinite(normal).all():
            owner.close()
            raise RuntimeError("cuDSS recovered a nonfinite panel displacement")
        self._rhs = None
        self._panel_solved = True
        return normal

    def original_residual(self):
        """Check the preceding elastic solve against the original matrix on the GPU."""
        owner = self.owner
        if not owner._workspace or self.generation != owner._generation or not (self._panel_solved or self._rhs is not None):
            raise RuntimeError("CUDA panel residual requires an elastic solve at the current frequency")
        if not hasattr(owner._lib, "gpu_panel_original_residual"):
            raise RuntimeError("The GPU panel library must be rebuilt to check its residual on the GPU")
        result = ctypes.c_double()
        rhs_ptr = self._rhs.ctypes.data if self._rhs is not None else None
        _check(owner._lib.gpu_panel_original_residual(owner._workspace, rhs_ptr,
                                                     owner._diagonal().ctypes.data, ctypes.byref(result)),
               "original equation check")
        return result.value

    @property
    def supports_normal_many(self):
        return hasattr(self.owner._lib, "gpu_panel_solve_normal_many")

    @property
    def supports_normal_many_device(self):
        return hasattr(self.owner._lib, "gpu_panel_solve_normal_many_device")

    def solve_normal_many(self, rhs, *, block_size=32):
        """Recover panel averages and check each original elastic equation."""
        if not self.supports_normal_many:
            raise RuntimeError("Rebuild the GPU panel library for blocked elastic recovery")
        rhs = np.asfortranarray(rhs, dtype=np.complex128)
        if rhs.ndim != 2 or rhs.shape[0] != self.size or not rhs.shape[1]:
            raise ValueError("Expected a nonempty matrix of original ice right-hand sides")
        return self._solve_normal_many(rhs.ctypes.data, rhs.shape[1], block_size,
                                       self.owner._lib.gpu_panel_solve_normal_many)

    def solve_normal_many_device(self, pointer, count, *, block_size=32):
        """Read the original ice RHS columns from a ready device buffer on this GPU."""
        if not self.supports_normal_many_device:
            raise RuntimeError("Rebuild the GPU panel library for device-resident elastic recovery")
        pointer, count = index(pointer), index(count)
        if pointer <= 0 or count < 1:
            raise ValueError("Expected a device address and at least one original ice right-hand side")
        return self._solve_normal_many(pointer, count, block_size,
                                       self.owner._lib.gpu_panel_solve_normal_many_device)

    def _solve_normal_many(self, pointer, count, block_size, solve):
        owner = self.owner
        if not owner._workspace or self.generation != owner._generation:
            raise RuntimeError("CUDA panel Schur factor is stale")
        block_size = index(block_size)
        if block_size < 1:
            raise ValueError("Elastic recovery block size must be positive")
        self._rhs = None
        self._panel_solved = False
        panels = len(self._scale)
        normal = np.empty((panels, count), dtype=np.complex128, order="F")
        residuals = np.empty(count)
        diagonal = owner._diagonal()
        stride = normal.itemsize
        for first in range(0, count, block_size):
            last = min(first + block_size, count)
            code = solve(owner._workspace, last - first, pointer + first * self.size * stride,
                         diagonal.ctypes.data, normal.ctypes.data + first * panels * stride,
                         residuals.ctypes.data + first * residuals.itemsize)
            if code:
                owner.close()
                _check(code, "blocked elastic recovery")
        normal *= self._scale[:, None]
        if not np.isfinite(normal).all() or not np.isfinite(residuals).all():
            owner.close()
            raise RuntimeError("cuDSS blocked elastic recovery returned a nonfinite value")
        return normal, residuals
