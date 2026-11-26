"""Optional complex symmetric MUMPS solver for the planar ice equations."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np


DEFAULT_LIBRARY = Path(__file__).with_name("libplanar_mumps.so")
ORDERINGS = {"auto": 7, "amd": 0, "scotch": 3, "pord": 4, "metis": 5}
INFO_FIELDS = (1, 2, 7, 16, 17, 20, 21, 22, 26, 29)


def build_library(mumps_root, output=DEFAULT_LIBRARY, *, openblas=None):
    """Use an isolated sequential MUMPS prefix with headers and libzmumps_seq."""
    prefix = Path(mumps_root).resolve()
    source = Path(__file__).with_suffix(".cpp")
    library = prefix / "lib" / "x86_64-linux-gnu"
    if not library.is_dir():
        library = prefix / "lib"
    libraries = sorted(library.glob("libzmumps_seq*.so"))
    if not libraries:
        raise FileNotFoundError(f"Could not find libzmumps_seq in {library}")
    output = Path(output).resolve()
    dependencies = [str(libraries[0])]
    library_dirs = [library]
    if openblas is not None:
        tuned = Path(openblas).resolve()
        dependencies[:0] = ["-Wl,--no-as-needed", str(tuned), "-Wl,--as-needed"]
        library_dirs.append(tuned.parent)
    subprocess.run([
        "c++", "-std=c++17", "-shared", "-fPIC", "-O3", str(source), f"-I{prefix / 'include'}",
        *dependencies, f"-Wl,--disable-new-dtags,-rpath,{os.pathsep.join(map(str, library_dirs))}", "-o", str(output),
    ], check=True)
    return output


@lru_cache(maxsize=2)
def _library(path):
    library = ctypes.CDLL(str(path))
    ptr, i32, i64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64
    library.planar_mumps_index_bytes.argtypes = ()
    library.planar_mumps_index_bytes.restype = ctypes.c_int
    library.planar_mumps_upper.argtypes = (i32, ctypes.c_int, ptr, ptr, ptr, i64, ptr, ptr, ptr, ctypes.c_int)
    library.planar_mumps_upper.restype = i64
    library.planar_mumps_create.argtypes = (i32, i64, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_char_p)
    library.planar_mumps_create.restype = ptr
    library.planar_mumps_phase.argtypes = (ptr, ctypes.c_int, ptr, ptr)
    library.planar_mumps_phase.restype = ctypes.c_int
    library.planar_mumps_solve.argtypes = (ptr, ptr, i32, ptr)
    library.planar_mumps_solve.restype = ctypes.c_int
    if hasattr(library, "planar_mumps_solve_sparse"):
        library.planar_mumps_solve_sparse.argtypes = (ptr, ptr, ptr, i32, ptr, ptr)
        library.planar_mumps_solve_sparse.restype = ctypes.c_int
    library.planar_mumps_destroy.argtypes = (ptr,)
    library.planar_mumps_destroy.restype = None
    return library


class PlanarMumps:
    """Factor the complex-symmetric ice operator, with no dense interface response."""

    def __init__(self, library_path=None, memory_mb=None, out_of_core=None, ordering="auto", verbose=False):
        self._handle = None
        self._numeric = False
        path = Path(library_path or DEFAULT_LIBRARY).resolve()
        self._lib = _library(path)
        self._dtype = np.dtype(f"int{8 * self._lib.planar_mumps_index_bytes()}")
        self._memory = int(memory_mb or 0)
        self._ooc = os.fsencode(Path(out_of_core).resolve()) if out_of_core else b""
        if self._memory < 0 or self._memory > np.iinfo(np.int32).max or len(self._ooc) > 255:
            raise ValueError("Invalid MUMPS working-memory limit or out-of-core directory")
        self._ordering = ORDERINGS[ordering]
        self._verbose = verbose
        self._n = None
        self._rows = self._cols = self._values = None
        self.info = {}

    def close(self):
        if self._handle is not None:
            self._lib.planar_mumps_destroy(self._handle)
            self._handle = None
        self._numeric = False
        self._n = None
        self._rows = self._cols = self._values = None

    def __del__(self):
        self.close()

    def _step(self, phase, call, *args):
        stats = np.empty(len(INFO_FIELDS), dtype=np.int64)
        now = perf_counter()
        status = call(self._handle, *args, stats.ctypes.data)
        details = {f"INFOG({i})": int(j) for i, j in zip(INFO_FIELDS, stats)}
        details["seconds"] = perf_counter() - now
        self.info[phase] = details
        if self._verbose and phase in ("analysis", "factor"):
            print(f"MUMPS {phase}: {details}", flush=True)
        if status < 0:
            raise RuntimeError(f"MUMPS {phase} failed: {details}")

    def factor(self, matrix, *, upper=False):
        """Read one triangle from a full or upper-only complex symmetric matrix."""
        matrix = matrix.tocsc(copy=False)
        n, m = matrix.shape
        if n != m or not 0 < n < np.iinfo(np.int32).max:
            raise ValueError("MUMPS requires a nonempty square matrix with 32-bit row IDs")
        matrix.sum_duplicates()
        matrix.sort_indices()
        self._numeric = False
        index_dtype = np.promote_types(matrix.indices.dtype, matrix.indptr.dtype)
        indices = matrix.indices.astype(index_dtype, copy=False)
        indptr = matrix.indptr.astype(index_dtype, copy=False)
        data = np.ascontiguousarray(matrix.data, dtype=np.complex128)
        common = (n, indices.dtype.itemsize, indptr.ctypes.data, indices.ctypes.data, data.ctypes.data)
        reuse = self._handle is not None and self._n == n
        if reuse:
            count = self._lib.planar_mumps_upper(
                *common, len(self._rows), self._rows.ctypes.data, self._cols.ctypes.data, self._values.ctypes.data, 1
            )
            reuse = count == len(self._rows)
            if reuse and upper and count != len(data):
                raise ValueError("Invalid upper-triangle format for MUMPS")
        if not reuse:
            self.close()
            nnz = self._lib.planar_mumps_upper(*common, 0, None, None, None, 0)
            if nnz < 1 or (upper and nnz != len(data)):
                raise ValueError("Invalid upper-triangle format for MUMPS")
            self._rows, self._cols = (np.empty(nnz, dtype=self._dtype) for _ in range(2))
            self._values = np.empty(nnz, dtype=np.complex128)
            copied = self._lib.planar_mumps_upper(
                *common, nnz, self._rows.ctypes.data, self._cols.ctypes.data, self._values.ctypes.data, 0
            )
            if copied != nnz:
                raise RuntimeError("Could not read the symmetric ice matrix")
            self._n = n
            self._handle = self._lib.planar_mumps_create(n, nnz, self._rows.ctypes.data, self._cols.ctypes.data,
                                                         self._ordering, self._memory, self._ooc)
            if self._handle is None:
                raise RuntimeError("Could not initialize MUMPS")
            try:
                self._step("analysis", self._lib.planar_mumps_phase, 1, None)
            except RuntimeError:
                self.close()
                raise
        self._step("factor", self._lib.planar_mumps_phase, 2, self._values.ctypes.data)
        self._numeric = True
        return self

    def solve(self, rhs, trans="N"):
        """Backsolve complex-symmetric A or its transpose with the current numeric factors."""
        if not self._numeric:
            raise RuntimeError("The current matrix has no numeric MUMPS factors")
        if trans not in {"N", "T", "H"}:
            raise ValueError("trans must be N, T, or H")
        values = np.array(rhs, dtype=np.complex128, order="F", copy=True)
        if values.ndim not in (1, 2) or values.shape[0] != self._n:
            raise ValueError("Wrong MUMPS right-hand-side shape")
        count = 1 if values.ndim == 1 else values.shape[1]
        if count >= np.iinfo(np.int32).max:
            raise ValueError("Too many right-hand sides for MUMPS")
        if count == 0:
            return values
        if trans == "H":
            np.conjugate(values, out=values)
        self._step("solve", self._lib.planar_mumps_solve, values.ctypes.data, count)
        if trans == "H":
            np.conjugate(values, out=values)
        return values

    def solve_sparse(self, indices, values):
        """Solve one ice load with only the supplied zero-based rows present."""
        if not self._numeric:
            raise RuntimeError("The current matrix has no numeric MUMPS factors")
        indices = np.asarray(indices)
        values = np.asarray(values)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or values.shape != indices.shape
                or indices.size >= np.iinfo(np.int32).max):
            raise ValueError("A sparse ice load needs matching integer indices and values")
        if indices.size == 0:
            return np.zeros(self._n, dtype=np.complex128)
        if np.any(indices < 0) or np.any(indices >= self._n) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError("Sparse ice load indices must be strictly increasing and in range")
        if not hasattr(self._lib, "planar_mumps_solve_sparse"):
            dense = np.zeros(self._n, dtype=np.complex128)
            dense[indices] = values
            return self.solve(dense)
        one_based = np.ascontiguousarray(indices, dtype=self._dtype) + 1
        values = np.array(values, dtype=np.complex128, copy=True)
        solution = np.empty(self._n, dtype=np.complex128)
        self._step("sparse_solve", self._lib.planar_mumps_solve_sparse,
                   one_based.ctypes.data, values.ctypes.data, len(values), solution.ctypes.data)
        return solution
