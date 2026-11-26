"""Optional persistent GPU Schur condensation for the elastic ice trace."""

import ctypes
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from scipy.sparse import triu


DEFAULT_LIBRARY = Path(__file__).with_name("libgpu_trace_factor.so")


def build_library(cudss_root, *, cuda_root=None, output=DEFAULT_LIBRARY):
    cuda_root = Path(cuda_root or os.environ.get('CUDA_ROOT', '/usr/local/cuda'))
    cudss_root = Path(cudss_root)
    source = Path(__file__).with_suffix(".cpp")
    cuda_lib = cuda_root / "targets/x86_64-linux/lib"
    subprocess.run([
        "c++", "-std=c++17", "-shared", "-fPIC", "-O3", str(source),
        f"-I{cuda_root / 'include'}", f"-I{cudss_root / 'include'}",
        f"-L{cuda_lib}", f"-L{cudss_root / 'lib'}", "-lcudart", "-lcudss", "-lcusolver", "-o", str(output),
    ], check=True)
    return Path(output).resolve()


@lru_cache(maxsize=2)
def _library(path):
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    try:
        library = ctypes.CDLL(str(path))
        library.gpu_trace_create.argtypes = (integer, integer, integer, integer, integer,
                                             pointer, pointer, pointer, pointer, ctypes.POINTER(pointer))
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"Unable to load the GPU trace solver {path}: {error}") from error
    library.gpu_trace_create.restype = integer
    library.gpu_trace_destroy.argtypes = (pointer,)
    library.gpu_trace_destroy.restype = None
    library.gpu_trace_factor.argtypes = (pointer, integer, pointer, pointer, pointer)
    library.gpu_trace_factor.restype = integer
    library.gpu_trace_solve.argtypes = (pointer, pointer, pointer)
    library.gpu_trace_solve.restype = integer
    return library


class GpuTraceFactor:
    """One reusable factor per worker. Numerical factors last one frequency."""

    def __init__(self, ice, *, device=0, library_path=None, lock=None, cached_matrix=False):
        self.ice = ice
        self.device = device
        self.library_path = Path(library_path or DEFAULT_LIBRARY).resolve()
        self._lock = lock if lock is not None else nullcontext()
        self._library = None
        self._workspace = ctypes.c_void_p()
        self._signature = None
        self._generation = 0
        self.timing = None
        self._use_cached_matrix = cached_matrix
        self._matrix_cache = None

    @contextmanager
    def _device_access(self):
        start = perf_counter()
        with self._lock:
            acquired = perf_counter()
            try:
                yield
            finally:
                if self.timing is not None:
                    self.timing["gpu_ice_wait_s"] = self.timing.get("gpu_ice_wait_s", 0.0) + acquired - start
                    self.timing["gpu_ice_native_s"] = self.timing.get("gpu_ice_native_s", 0.0) + perf_counter() - acquired

    def close(self):
        if self._workspace:
            with self._device_access():
                self._library.gpu_trace_destroy(self._workspace)
            self._workspace = ctypes.c_void_p()
            self._signature = None

    def __del__(self):
        self.close()

    def factor(self, s, load, average, *, matrix=None, upper_csr=None):
        """Build FP64 numeric factors while reusing an identical CSR structure."""
        self._generation += 1
        if upper_csr is not None:
            if matrix is None:
                raise ValueError("The original matrix is required with a prepacked GPU upper triangle")
            rows, columns, values = upper_csr
            if rows.dtype != np.int32 or columns.dtype != np.int32 or values.dtype != np.complex128:
                raise ValueError("The GPU Schur factor requires complex128 values and 32-bit CSR indices")
        elif matrix is None and self._use_cached_matrix:
            if self._matrix_cache is None:
                from .ice_matrix import GpuIceMatrix

                self._matrix_cache = GpuIceMatrix(self.ice)
            matrix, (rows, columns, values) = self._matrix_cache.assemble(s)
        else:
            if matrix is None:
                matrix = self.ice.dynamic_matrix(s)
            upper = triu(matrix, format="csr")
            upper.sum_duplicates()
            upper.sort_indices()
            if max(matrix.shape[0], upper.nnz) >= np.iinfo(np.int32).max:
                raise RuntimeError("The GPU Schur factor uses 32-bit CSR indices")
            rows = upper.indptr.astype(np.int32, copy=False)
            columns = upper.indices.astype(np.int32, copy=False)
            values = upper.data.astype(np.complex128, copy=False)
        n = matrix.shape[0]
        trace = np.unique(load.nonzero()[0])
        if not 0 < trace.size < n:
            raise RuntimeError("GPU ice solve requires proper nonempty wet trace and interior sets")
        avg_columns = np.unique(average.nonzero()[1])
        if not np.array_equal(trace, avg_columns):
            raise RuntimeError("The normal displacement must be supported by the wetted trace")

        candidate = (rows, columns, trace, load.shape[1])
        old = self._signature
        reused_analysis = (old is not None and old[3] == candidate[3]
                           and all(np.array_equal(a, b) for a, b in zip(old[:3], candidate[:3])))
        if self.timing is not None:
            self.timing["ice_gpu_symbolic_reused"] = int(reused_analysis)
        if not reused_analysis:
            analysis_started = perf_counter()
            self.close()
            self._library = _library(self.library_path)
            marks = np.zeros(n, dtype=np.int32)
            marks[trace] = 1
            with self._device_access():
                code = self._library.gpu_trace_create(
                    self.device, n, values.size, trace.size, load.shape[1], rows.ctypes.data,
                    columns.ctypes.data, values.ctypes.data, marks.ctypes.data, ctypes.byref(self._workspace),
                )
            if code:
                raise RuntimeError(f"cuDSS symbolic Schur analysis failed with code {code}")
            self._signature = candidate
            if self.timing is not None:
                self.timing["ice_gpu_symbolic_s"] = perf_counter() - analysis_started

        load_values = np.asfortranarray(load[trace].toarray(), dtype=np.complex128)
        trace_values = np.empty_like(load_values, order="F")
        numeric_started = perf_counter()
        with self._device_access():
            code = self._library.gpu_trace_factor(self._workspace, values.size, values.ctypes.data,
                                                  load_values.ctypes.data, trace_values.ctypes.data)
        if code or not np.isfinite(trace_values).all():
            self.close()
            raise RuntimeError(f"GPU ice numeric Schur factorization failed with code {code}")
        response = average[:, trace] @ trace_values
        if self.timing is not None:
            self.timing["ice_gpu_numeric_and_response_s"] = perf_counter() - numeric_started
        return _GpuNumericFactor(self, self._generation, matrix, response)


class _GpuNumericFactor:
    def __init__(self, owner, generation, matrix, response):
        self.owner = owner
        self.generation = generation
        self.matrix = matrix
        self.response = response

    def panel_response(self):
        return self.response

    def solve(self, rhs):
        """Solve the original sparse system using cuDSS forward/back substitution."""
        owner = self.owner
        if not owner._workspace or self.generation != owner._generation:
            raise RuntimeError("GPU ice factor is stale")
        rhs = np.ascontiguousarray(rhs, dtype=np.complex128)
        if rhs.shape != (self.matrix.shape[0],):
            raise ValueError("Expected one original ice right-hand side")
        solution = np.empty_like(rhs)
        with owner._device_access():
            code = owner._library.gpu_trace_solve(owner._workspace, rhs.ctypes.data, solution.ctypes.data)
        if code or not np.isfinite(solution).all():
            owner.close()
            raise RuntimeError(f"GPU ice sparse backsolve failed with code {code}")
        return solution

    def relative_residual(self, solution, rhs):
        error = np.linalg.norm(self.matrix @ solution - rhs)
        norm = np.linalg.norm(rhs)
        return float(error / norm if norm else error)
