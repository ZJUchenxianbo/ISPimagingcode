"""cuDSS complex-double ice factorization and repeated single-RHS solves."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np


DEFAULT_LIBRARY = Path(__file__).with_name("libplanar_cudss.so")


def build_library(cudss_root, cuda_root=None, output=DEFAULT_LIBRARY):
    """Build against a cuDSS 0.7.1 runtime and its matching CUDA toolkit."""
    dss = Path(cudss_root).resolve()
    cuda = Path(cuda_root or os.environ.get('CUDA_ROOT', '/usr/local/cuda')).resolve()
    libs = (dss / "lib", (cuda / "lib64").resolve())
    source = Path(__file__).with_suffix(".cpp")
    output = Path(output).resolve()
    subprocess.run([
        "c++", "-std=c++17", "-O3", "-fPIC", "-shared", str(source),
        f"-I{dss / 'include'}", f"-I{cuda / 'include'}",
        *(f"-L{directory}" for directory in libs), "-lcudss", "-lcudart",
        f"-Wl,-rpath,{os.pathsep.join(map(str, libs))}", "-o", str(output),
    ], check=True)
    return output


@lru_cache(maxsize=2)
def _library(path):
    library = ctypes.CDLL(str(path))
    ptr, cint, i64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    library.planar_cudss_upper.argtypes = (i64, cint, ptr, ptr, ptr, i64, ptr, ptr, ptr, cint)
    library.planar_cudss_upper.restype = i64
    library.planar_cudss_create.argtypes = (cint, cint, i64, i64, ptr, ptr, ptr, cint, i64, cint, ptr, ctypes.POINTER(ptr))
    library.planar_cudss_create.restype = cint
    library.planar_cudss_factor.argtypes = (ptr, ptr, ptr)
    library.planar_cudss_factor.restype = cint
    library.planar_cudss_solve.argtypes = (ptr, ptr, ptr)
    library.planar_cudss_solve.restype = cint
    library.planar_cudss_destroy.argtypes = (ptr,)
    library.planar_cudss_destroy.restype = None
    return library


class PlanarCudss:
    """Reuse the analysis of a complex symmetric sparse ice matrix on a CUDA device."""

    def __init__(self, library_path=None, device=1, verbose=False, *,
                 hybrid_memory=False, memory_limit=None, refinement_steps=1):
        self._workspace = ctypes.c_void_p()
        self._close_state()
        self._lib = _library(Path(library_path or DEFAULT_LIBRARY).resolve())
        self.device = int(device)
        self.verbose = verbose
        self.hybrid_memory = bool(hybrid_memory)
        self.memory_limit = int(memory_limit or 0)
        self.refinement_steps = int(refinement_steps)
        if min(self.device, self.memory_limit, self.refinement_steps) < 0:
            raise ValueError("Invalid CUDA device, memory limit, or refinement steps")
        if self.memory_limit and not self.hybrid_memory:
            raise ValueError("A cuDSS device-memory limit requires hybrid memory")
        self.info = {}

    def _close_state(self):
        self._n = self._offsets = self._columns = self._values = None
        self._numeric = False

    def close(self):
        if self._workspace:
            self._lib.planar_cudss_destroy(self._workspace)
            self._workspace = ctypes.c_void_p()
        self._close_state()

    def __del__(self):
        self.close()

    @staticmethod
    def _check(status, phase):
        if status:
            raise RuntimeError(f"CUDA cuDSS {phase} failed (native status {status})")

    def factor(self, matrix, *, upper=False):
        """Factor A, given its full complex-symmetric CSR or upper-only CSR."""
        matrix = matrix.tocsr(copy=False)
        n, m = matrix.shape
        if n != m or not 0 < n < np.iinfo(np.int32).max:
            raise ValueError("cuDSS requires a nonempty square matrix with 32-bit row IDs")
        if not matrix.nnz:
            raise ValueError("cuDSS requires a sparse matrix with entries")
        matrix.sum_duplicates()
        matrix.sort_indices()
        self._numeric = False
        dtype = np.promote_types(matrix.indptr.dtype, matrix.indices.dtype)
        offsets = matrix.indptr.astype(dtype, copy=False)
        cols = matrix.indices.astype(dtype, copy=False)
        values = np.ascontiguousarray(matrix.data, dtype=np.complex128)
        args = (n, dtype.itemsize, offsets.ctypes.data, cols.ctypes.data, values.ctypes.data)
        start = perf_counter()
        new_analysis = self._n != n or self._offsets.dtype != dtype
        if not new_analysis:
            count = self._lib.planar_cudss_upper(
                *args, self._columns.size, self._offsets.ctypes.data,
                self._columns.ctypes.data, self._values.ctypes.data, 1,
            )
            new_analysis = count != self._columns.size
        if new_analysis:
            self.close()
            count = self._lib.planar_cudss_upper(*args, 0, None, None, None, 0)
            if count < 1 or (upper and count != matrix.nnz):
                raise ValueError("Invalid upper triangle given to cuDSS")
            self._offsets = np.empty(n + 1, dtype=dtype)
            self._columns = np.empty(count, dtype=dtype)
            self._values = np.empty(count, dtype=np.complex128)
            copied = self._lib.planar_cudss_upper(
                *args, count, self._offsets.ctypes.data, self._columns.ctypes.data, self._values.ctypes.data, 0,
            )
            if copied != count:
                raise RuntimeError("Failed to extract the upper triangle for cuDSS")
        elif upper and count != matrix.nnz:
            raise ValueError("Invalid upper triangle given to cuDSS")
        self.info["pack_upper_s"] = perf_counter() - start
        if new_analysis:
            start = perf_counter()
            memory = np.empty(18, dtype=np.int64)
            code = self._lib.planar_cudss_create(
                self.device, dtype.itemsize, n, count, self._offsets.ctypes.data,
                self._columns.ctypes.data, self._values.ctypes.data, self.hybrid_memory,
                self.memory_limit, self.refinement_steps, memory.ctypes.data, ctypes.byref(self._workspace),
            )
            self._check(code, "analysis")
            self._n = n
            self.info.update(analysis_s=perf_counter()-start, dimension=n, upper_nnz=count,
                             device_index=self.device, device_memory_peak_estimate=int(memory[1]),
                             host_memory_peak_estimate=int(memory[3]),
                             device_free_after_analysis=int(memory[16]))
            if self.verbose:
                print(f"cuDSS analysis: {self.info}", flush=True)
        start = perf_counter()
        stats = np.empty(3, dtype=np.int64)
        updated = None if new_analysis else self._values.ctypes.data
        self._check(self._lib.planar_cudss_factor(self._workspace, updated, stats.ctypes.data), "factor")
        self.info.update(factor_s=perf_counter()-start, reused_analysis=not new_analysis,
                         lu_nnz=int(stats[0]), device_free_after_factor=int(stats[1]),
                         device_total=int(stats[2]))
        self._numeric = True
        if self.verbose:
            print(f"cuDSS factor: {self.info}", flush=True)
        return self

    def solve(self, rhs, trans="N"):
        if not self._numeric:
            raise RuntimeError("The current matrix has no numeric cuDSS factors")
        if trans not in ("N", "T", "H"):
            raise ValueError("trans must be N, T, or H")
        rhs = np.ascontiguousarray(rhs, dtype=np.complex128)
        if rhs.shape != (self._n,):
            raise ValueError("cuDSS requires one right-hand-side vector")
        if trans == "H":
            rhs = np.conjugate(rhs)
        result = np.empty_like(rhs)
        start = perf_counter()
        code = self._lib.planar_cudss_solve(self._workspace, rhs.ctypes.data, result.ctypes.data)
        self._check(code, "solve")
        self.info["solve_s"] = perf_counter() - start
        if trans == "H":
            np.conjugate(result, out=result)
        return result
