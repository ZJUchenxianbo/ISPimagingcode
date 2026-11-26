"""Keep planar FP64 water boundary operators and LU factors on one CUDA device."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator


@lru_cache(maxsize=2)
def _load(path):
    lib = ctypes.CDLL(str(path))
    ptr = ctypes.c_void_p
    integer = ctypes.c_int
    lib.planar_water_create.argtypes = (integer, integer, integer, ptr, ptr, ctypes.POINTER(ptr))
    lib.planar_water_create.restype = integer
    lib.planar_water_destroy.argtypes = (ptr,)
    lib.planar_water_destroy.restype = None
    lib.planar_water_action.argtypes = (ptr, ptr, ptr, ptr)
    lib.planar_water_action.restype = integer
    lib.planar_water_factor.argtypes = (ptr, ptr)
    lib.planar_water_factor.restype = integer
    lib.planar_water_precondition.argtypes = (ptr, ptr, ptr)
    lib.planar_water_precondition.restype = integer
    return lib


@lru_cache(maxsize=2)
def _load_direct(path):
    lib = _load(path)
    try:
        direct = lib.planar_water_direct
    except AttributeError as error:
        raise RuntimeError(f"Rebuild {path} to enable the coupled CUDA direct solve") from error
    ptr = ctypes.c_void_p
    direct.argtypes = (ctypes.c_int,) * 3 + (ptr,) * 5
    direct.restype = ctypes.c_int
    return direct


@lru_cache(maxsize=2)
def _load_workspace(path):
    lib = _load(path)
    ptr = ctypes.c_void_p
    try:
        lib.planar_water_direct_create.argtypes = (ctypes.c_int,) * 3 + (ctypes.POINTER(ptr),)
        lib.planar_water_direct_create.restype = ctypes.c_int
        lib.planar_water_direct_destroy.argtypes = (ptr,)
        lib.planar_water_direct_destroy.restype = None
        lib.planar_water_direct_solve.argtypes = (ptr,) * 6
        lib.planar_water_direct_solve.restype = ctypes.c_int
        lib.planar_water_direct_resident.argtypes = (ptr,) * 7
        lib.planar_water_direct_resident.restype = ctypes.c_int
        lib.planar_water_direct_residual.argtypes = (ptr,) * 6 + (ctypes.POINTER(ctypes.c_double),)
        lib.planar_water_direct_residual.restype = ctypes.c_int
    except AttributeError as error:
        raise RuntimeError(f"Rebuild {path} to enable reusable CUDA water workspaces") from error
    return lib


@lru_cache(maxsize=2)
def _load_refinement(path):
    lib = _load_workspace(path)
    ptr = ctypes.c_void_p
    integer = ctypes.c_int
    try:
        lib.planar_water_direct_create_mixed.argtypes = (integer,) * 3 + (ctypes.POINTER(ptr),)
        lib.planar_water_direct_create_mixed.restype = integer
        lib.planar_water_direct_refinement_stats.argtypes = (
            ptr, ctypes.POINTER(integer), ctypes.POINTER(integer), ctypes.POINTER(ctypes.c_double),
        )
        lib.planar_water_direct_refinement_stats.restype = None
    except AttributeError as error:
        raise RuntimeError(f"Rebuild {path} to enable CUDA water mixed-precision refinement") from error
    return lib


@lru_cache(maxsize=2)
def _load_fast_refinement(path):
    lib = _load_refinement(path)
    try:
        create = lib.planar_water_direct_create_fast
    except AttributeError as error:
        raise RuntimeError(f"Rebuild {path} with nvcc -x cu to enable fast CUDA water refinement") from error
    create.argtypes = (ctypes.c_int,) * 3 + (ctypes.POINTER(ctypes.c_void_p),)
    create.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=2)
def _load_gemm3m(path):
    lib = _load_workspace(path)
    try:
        setup = lib.planar_water_direct_use_gemm3m
    except AttributeError as error:
        raise RuntimeError(f"Rebuild {path} to enable CUDA water 3M matrix products") from error
    setup.argtypes = (ctypes.c_void_p,)
    setup.restype = None
    return setup


@lru_cache(maxsize=2)
def _load_prepared(path):
    lib = _load_workspace(path)
    ptr = ctypes.c_void_p
    integer = ctypes.c_int
    try:
        lib.planar_water_direct_solve_prepared.argtypes = (
            ptr, ptr, ptr, integer, ptr, ptr, ctypes.POINTER(ctypes.c_double),
        )
        lib.planar_water_direct_solve_prepared.restype = integer
        lib.planar_water_direct_coupling.argtypes = (ptr, ptr, integer, ptr, ptr, ptr)
        lib.planar_water_direct_coupling.restype = integer
    except AttributeError as error:
        raise RuntimeError(f"Rebuild {path} to reuse prepared CUDA water factors") from error
    if hasattr(lib, "planar_water_direct_solve_prepared_batch"):
        lib.planar_water_direct_solve_prepared_batch.argtypes = (
            ptr, ptr, ptr, integer, ptr, ptr, ptr, ctypes.POINTER(ctypes.c_double),
        )
        lib.planar_water_direct_solve_prepared_batch.restype = integer
    return lib


def _check(code, stage):
    if code:
        library = "CUDA"
        if code >= 30000:
            library, code = "cuSOLVER numeric", code - 30000
        elif code >= 20000:
            library, code = "cuSOLVER", code - 20000
        elif code >= 10000:
            library, code = "cuBLAS", code - 10000
        raise RuntimeError(f"{library} planar water {stage} failed (status {code})")


def solve_coupled_water_gpu(library, device, water, coupling, response, rhs):
    """Apply the ice panel response and solve the unchanged complex water system."""
    water, coupling, response, rhs = (
        np.ascontiguousarray(value, dtype=np.complex128) for value in (water, coupling, response, rhs)
    )
    if coupling.ndim != 2:
        raise ValueError("Water coupling must be a rectangular matrix")
    rows, wet = coupling.shape
    if not (0 < wet <= rows and water.shape == (rows, rows)
            and response.shape == (wet, wet) and rhs.shape == (rows,)):
        raise ValueError("Incompatible water, coupling, response, or right-hand-side dimensions")
    direct = _load_direct(Path(library).resolve())
    solution = np.empty(rows, dtype=np.complex128)
    _check(direct(device, rows, wet, water.ctypes.data, coupling.ctypes.data, response.ctypes.data,
                  rhs.ctypes.data, solution.ctypes.data), "coupled direct solve")
    return solution


class PlanarWaterDirect:
    """Reuse CUDA allocations and library handles across frequencies in one worker."""

    def __init__(self, library_path, device, rows, wet, *, mixed_precision=False, gemm3m=False,
                 fast_refinement=False):
        if not 0 < wet <= rows:
            raise ValueError("Invalid water and wet panel counts")
        if fast_refinement and (mixed_precision or gemm3m):
            raise ValueError("Fast refinement uses a separate provisional single-precision product and factor")
        path = Path(library_path).resolve()
        self.path = path
        self.mixed_precision = mixed_precision
        self.fast_refinement = fast_refinement
        if fast_refinement:
            self.lib = _load_fast_refinement(path)
            create = self.lib.planar_water_direct_create_fast
        elif mixed_precision:
            self.lib = _load_refinement(path)
            create = self.lib.planar_water_direct_create_mixed
        else:
            self.lib = _load_workspace(path)
            create = self.lib.planar_water_direct_create
        self.rows, self.wet = rows, wet
        self.ptr = ctypes.c_void_p()
        _check(create(device, rows, wet, ctypes.byref(self.ptr)), "workspace allocation")
        if gemm3m:
            _load_gemm3m(path)(self.ptr)

    def close(self):
        if getattr(self, "ptr", None):
            self.lib.planar_water_direct_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _workspace(self):
        if not self.ptr:
            raise RuntimeError("The CUDA water workspace is closed")
        return self.ptr

    def _inputs(self, response, rhs):
        response = np.ascontiguousarray(response, dtype=np.complex128)
        rhs = np.ascontiguousarray(rhs, dtype=np.complex128)
        if response.shape != (self.wet, self.wet) or rhs.shape != (self.rows,):
            raise ValueError("Incompatible ice panel response or water right-hand-side dimensions")
        return response, rhs

    def solve(self, water, coupling, response, rhs):
        """Solve with NumPy inputs stored on the host."""
        if self.fast_refinement:
            raise ValueError("Fast CUDA water refinement requires device-resident BEM matrices")
        workspace = self._workspace()
        water = np.ascontiguousarray(water, dtype=np.complex128)
        coupling = np.ascontiguousarray(coupling, dtype=np.complex128)
        if water.shape != (self.rows, self.rows) or coupling.shape != (self.rows, self.wet):
            raise ValueError("Incompatible water boundary or ice coupling dimensions")
        response, rhs = self._inputs(response, rhs)
        solution = np.empty(self.rows, dtype=np.complex128)
        _check(self.lib.planar_water_direct_solve(workspace, water.ctypes.data, coupling.ctypes.data,
                                                 response.ctypes.data, rhs.ctypes.data, solution.ctypes.data),
               "coupled direct solve")
        return solution

    def solve_resident(self, boundary_device, coupling_device, response, rhs, *, ready_event=0):
        """Solve with contiguous row-major complex128 A and C on this workspace's device."""
        workspace = self._workspace()
        if not boundary_device or not coupling_device:
            raise ValueError("CUDA water boundary and ice coupling pointers are required")
        response, rhs = self._inputs(response, rhs)
        solution = np.empty(self.rows, dtype=np.complex128)
        _check(self.lib.planar_water_direct_resident(workspace, boundary_device, coupling_device,
                                                     response.ctypes.data, rhs.ctypes.data, solution.ctypes.data,
                                                     ready_event), "resident coupled direct solve")
        return solution

    @staticmethod
    def _columns(values, rows, name):
        columns = np.asfortranarray(values, dtype=np.complex128)
        if columns.ndim == 1 and columns.shape == (rows,):
            return columns, 1
        if columns.ndim == 2 and columns.shape[0] == rows and 0 < columns.shape[1] <= np.iinfo(np.int32).max:
            return columns, columns.shape[1]
        raise ValueError(f"{name} must have shape ({rows},) or ({rows}, number of columns)")

    def solve_prepared_resident(self, boundary_device, coupling_device, rhs, *, normal=None, return_residual=False):
        """Solve one or more RHS with the latest resident Schur operator and retained factors.

        Both original device blocks must remain unmodified and alive through this call.
        If supplied, independently add the ice contribution C @ normal to each RHS.
        RHS columns are certified against the original complex128 Schur operator.
        """
        workspace = self._workspace()
        if not boundary_device or not coupling_device:
            raise ValueError("CUDA water boundary and ice coupling pointers are required")
        rhs, count = self._columns(rhs, self.rows, "Water right-hand side")
        if normal is not None:
            normal, normal_count = self._columns(normal, self.wet, "Normal displacement")
            if normal_count != count:
                raise ValueError("Provide one normal displacement column per water right-hand side")
        solution = np.empty_like(rhs, order="F")
        residuals = np.empty(count, dtype=np.float64)
        residual_pointer = residuals.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib = _load_prepared(self.path)
        batch = getattr(lib, "planar_water_direct_solve_prepared_batch", None)
        if count > 1 and batch is not None and os.environ.get("PLANAR_BATCH_WATER_TANGENT", "1") != "0":
            status = batch(workspace, boundary_device, coupling_device, count, rhs.ctypes.data,
                           None if normal is None else normal.ctypes.data, solution.ctypes.data, residual_pointer)
        else:
            if normal is not None:
                correction = self.coupling_resident(coupling_device, normal)
                rhs = np.asfortranarray(rhs + correction.reshape(rhs.shape))
            status = lib.planar_water_direct_solve_prepared(
                workspace, boundary_device, coupling_device, count, rhs.ctypes.data, solution.ctypes.data,
                residual_pointer)
        _check(status, "prepared resident solve or FP64 residual certification")
        if return_residual:
            return solution, float(residuals[0]) if rhs.ndim == 1 else residuals
        return solution

    def coupling_resident(self, coupling_device, normal, *, ready_event=0):
        """Apply the original complex128 device coupling C to normal-displacement columns."""
        workspace = self._workspace()
        if not coupling_device:
            raise ValueError("A CUDA ice coupling pointer is required")
        normal, count = self._columns(normal, self.wet, "Normal displacement")
        shape = (self.rows,) if normal.ndim == 1 else (self.rows, count)
        product = np.empty(shape, dtype=np.complex128, order="F")
        lib = _load_prepared(self.path)
        _check(lib.planar_water_direct_coupling(workspace, coupling_device, count, normal.ctypes.data,
                                                product.ctypes.data, ready_event), "resident ice coupling product")
        return product

    def residual_resident(self, boundary_device, coupling_device, solution, normal, rhs):
        """Compute ||A p − C u_n − b|| / ||b|| against the original device blocks."""
        workspace = self._workspace()
        if not boundary_device or not coupling_device:
            raise ValueError("CUDA water boundary and ice coupling pointers are required")
        solution, normal, rhs = (np.ascontiguousarray(value, dtype=np.complex128)
                                 for value in (solution, normal, rhs))
        if solution.shape != (self.rows,) or normal.shape != (self.wet,) or rhs.shape != (self.rows,):
            raise ValueError("Incompatible pressure, normal displacement, or right-hand-side dimensions")
        relative = ctypes.c_double()
        _check(self.lib.planar_water_direct_residual(workspace, boundary_device, coupling_device,
                                                     solution.ctypes.data, normal.ctypes.data, rhs.ctypes.data,
                                                     ctypes.byref(relative)), "resident original-system residual")
        return relative.value

    @property
    def refinement_stats(self):
        """Acceptance, steps, and FP64 relative Schur residual of the latest attempt."""
        workspace = self._workspace()
        if not (self.mixed_precision or self.fast_refinement):
            return False, None, None
        accepted = ctypes.c_int()
        steps = ctypes.c_int()
        relative = ctypes.c_double()
        self.lib.planar_water_direct_refinement_stats(
            workspace, ctypes.byref(accepted), ctypes.byref(steps), ctypes.byref(relative),
        )
        return bool(accepted.value), steps.value, relative.value if np.isfinite(relative.value) else None


class WaterCUDA:
    """Transfer the matrices once, with only vectors crossing PCIe in GMRES."""

    def __init__(self, library, device, boundary, coupling):
        self.lib = _load(Path(library).resolve())
        self.ptr = ctypes.c_void_p()
        self.rows, self.wet = coupling.shape
        self.device = device
        boundary = np.ascontiguousarray(boundary, dtype=np.complex128)
        coupling = np.ascontiguousarray(coupling, dtype=np.complex128)
        _check(self.lib.planar_water_create(device, self.rows, self.wet, boundary.ctypes.data,
                                            coupling.ctypes.data, ctypes.byref(self.ptr)), "allocation/upload")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.lib.planar_water_destroy(self.ptr)
        self.ptr = None

    def action(self, vector, normal):
        vector = np.ascontiguousarray(vector, dtype=np.complex128)
        normal = np.ascontiguousarray(normal, dtype=np.complex128)
        out = np.empty(self.rows, dtype=np.complex128)
        _check(self.lib.planar_water_action(self.ptr, vector.ctypes.data, normal.ctypes.data, out.ctypes.data),
               "matrix-vector product")
        return out

    def factor(self, matrix):
        matrix = np.asfortranarray(matrix, dtype=np.complex128)
        _check(self.lib.planar_water_factor(self.ptr, matrix.ctypes.data), "preconditioner factorization")
        return LinearOperator((self.rows, self.rows), matvec=self.solve, dtype=np.complex128)

    def solve(self, vector):
        vector = np.ascontiguousarray(vector, dtype=np.complex128)
        out = np.empty(self.rows, dtype=np.complex128)
        _check(self.lib.planar_water_precondition(self.ptr, vector.ctypes.data, out.ctypes.data),
               "preconditioner triangular solve")
        return out
