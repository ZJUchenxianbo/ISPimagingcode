"""Constant-panel Cartesian 2D acoustic BEM on a complex-stretched contour."""

import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import kv, roots_legendre


@lru_cache(maxsize=None)
def _quadrature(order):
    points, weights = roots_legendre(order)
    return (points + 1.0) / 2.0, weights / 2.0


@dataclass(frozen=True)
class PlanarStretch:
    """Quadratic damping outside the lateral and vertical physical limits."""

    xmin: float
    xmax: float
    zlimit: float
    pml_x: float
    pml_z: float
    damping_x: float
    damping_z: float
    alpha: float
    z_direction: int = -1

    def __post_init__(self):
        parameters = (self.xmin, self.xmax, self.zlimit, self.pml_x, self.pml_z,
                      self.damping_x, self.damping_z, self.alpha)
        if (not np.isfinite(parameters).all() or self.xmax <= self.xmin
                or min(parameters[3:]) < 0 or self.z_direction not in (-1, 1)):
            raise ValueError("Invalid Cartesian PML geometry or damping")

    def _extension(self, distance, width, damping, s):
        ext = np.zeros_like(distance, dtype=np.complex128)
        derivative = np.ones_like(ext)
        if width == 0 or damping == 0:
            return ext, derivative
        active = distance > 0
        fraction = distance[active] / width
        nodes, weights = _quadrature(16)
        clipped = np.minimum(fraction, 1.0)
        q = clipped[:, None] * nodes
        extension = clipped * np.sum(weights * damping * q**2 / (s + self.alpha * (1.0 - q)), axis=-1)
        ext[active] = width * (extension + (fraction - clipped) * damping / s)
        derivative[active] += damping * clipped**2 / (s + self.alpha * (1.0 - clipped))
        return ext, derivative

    def horizontal(self, x, s):
        x = np.asarray(x, dtype=float)
        distance = np.maximum(self.xmin - x, x - self.xmax)
        ext, derivative = self._extension(distance, self.pml_x, self.damping_x, s)
        return x + np.where(x < self.xmin, -ext, ext), derivative

    def vertical(self, z, s):
        z = np.asarray(z, dtype=float)
        ext, derivative = self._extension(self.z_direction * (z - self.zlimit), self.pml_z, self.damping_z, s)
        return z + self.z_direction * ext, derivative


def split_edges(start, stop, spacing, *splits):
    """Split at physical/PML joins, then keep every panel below the given length."""
    breaks = [start, *(x for x in sorted(set(splits), reverse=stop < start)
                        if min(start, stop) < x < max(start, stop)), stop]
    parts = [np.linspace(a, b, int(np.ceil(abs(b - a) / spacing)) + 1)
             for a, b in zip(breaks[:-1], breaks[1:])]
    return np.concatenate([parts[0], *(part[1:] for part in parts[1:])])


def _fixed_wall_edges(surface, outer, join, reference, spacing, *, reverse=False):
    """Connect the moving interface to the outer wall at fixed panel counts."""
    has_join = min(reference, outer) < join < max(reference, outer)
    if has_join != (min(surface, outer) < join < max(surface, outer)) or (
        reference == outer or (surface < outer) != (reference < outer)
    ):
        raise ValueError("The interface must stay on the reference side of the outer wall and PML join")
    stop = join if has_join else outer
    count = int(np.ceil(abs(stop - reference) / spacing))
    moving = np.linspace(stop, surface, count + 1) if reverse else np.linspace(surface, stop, count + 1)
    if has_join:
        if reverse:
            return np.r_[split_edges(outer, join, spacing)[:-1], moving]
        return np.r_[moving, split_edges(join, outer, spacing)[1:]]
    return moving


def free_green(s, distance, sound_speed):
    """Outgoing 2D Laplace Green function for a unit point source."""
    return kv(0, (s / sound_speed) * distance) / (2.0 * np.pi)


@lru_cache(maxsize=1)
def build_planar_library():
    """Load optional large-argument acceleration; SciPy also works by itself."""
    source = Path(__file__).with_name("kernels.cpp")
    output = source.with_name("_planar_kernels.so")
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    try:
        if not output.exists() or output.stat().st_mtime_ns < source.stat().st_mtime_ns:
            try:
                subprocess.run(["c++", "-std=c++17", "-O3", "-fPIC", "-shared", str(source), "-o", str(temporary)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        library = ctypes.CDLL(str(output))
    except (OSError, subprocess.CalledProcessError):
        return None
    ptr = ctypes.c_void_p
    double = ctypes.c_double
    library.planar_accumulate_regular.argtypes = (ctypes.c_int,) * 4 + (double, double) + (ptr,) * 10
    library.planar_accumulate_regular.restype = ptr
    library.planar_near_fields.argtypes = (ptr,) + (ctypes.POINTER(ptr),) * 4
    library.planar_near_fields.restype = ctypes.c_size_t
    library.planar_free_near_fields.argtypes = (ptr,)
    library.planar_free_near_fields.restype = None
    return library


def _native_block(library, wave, geometry, single, double):
    n_targets, n_panels = single.shape
    double_columns = double.shape[1]
    native = library.planar_accumulate_regular(n_targets, n_panels, len(geometry[-1]), double_columns,
                                               wave.real, wave.imag, *(x.ctypes.data for x in geometry),
                                               single.ctypes.data, double.ctypes.data)
    if not native:
        raise RuntimeError("Native planar acoustic kernel could not allocate work arrays")
    try:
        pointers = [ctypes.c_void_p() for _ in range(4)]
        size = library.planar_near_fields(native, *(ctypes.byref(ptr) for ptr in pointers))
        if size:
            indices = np.ctypeslib.as_array(ctypes.cast(pointers[0], ctypes.POINTER(ctypes.c_int32)), shape=(size,))
            arguments, coeff_s, coeff_k = (np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double)),
                                                               shape=(2 * size,)).view(np.complex128)
                                           for ptr in pointers[1:])
            _complete_regular(indices, arguments, coeff_s, coeff_k, single, double)
    finally:
        library.planar_free_near_fields(native)


def _complete_regular(indices, arguments, coeff_s, coeff_k, single, double):
    np.add.at(single.ravel(), indices, coeff_s * kv(0, arguments))
    has_double = coeff_k != 0
    rows, columns = np.divmod(indices[has_double], single.shape[1])
    np.add.at(double, (rows, columns), coeff_k[has_double] * kv(1, arguments[has_double]))


@lru_cache(maxsize=None)
def _load_cuda_assembly(path):
    library = ctypes.CDLL(str(path))
    ptr = ctypes.c_void_p
    library.planar_cuda_create.argtypes = (ctypes.c_int,) * 5 + (ptr,) * 6 + (ctypes.POINTER(ptr),)
    library.planar_cuda_create.restype = ctypes.c_int
    library.planar_cuda_rows.argtypes = (ptr, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                         ptr, ptr, ptr, ptr, ctypes.POINTER(ptr), ctypes.POINTER(ctypes.c_int))
    library.planar_cuda_rows.restype = ctypes.c_int
    library.planar_cuda_destroy.argtypes = (ptr,)
    library.planar_cuda_destroy.restype = None
    library.planar_cuda_error.argtypes = (ctypes.c_int,)
    library.planar_cuda_error.restype = ctypes.c_char_p
    if hasattr(library, "planar_cuda_prepare"):
        library.planar_cuda_prepare.argtypes = (ptr, ctypes.c_int, ctypes.c_int, ptr, ptr, ctypes.c_int,
                                               *(ctypes.c_double,) * 4, ctypes.c_int, ctypes.c_int)
        library.planar_cuda_prepare.restype = ctypes.c_int
        library.planar_cuda_rows_v2.argtypes = (ptr, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                               ptr, ptr, ptr, ptr, ctypes.POINTER(ptr), ctypes.POINTER(ctypes.c_int),
                                               ctypes.POINTER(ctypes.c_double))
        library.planar_cuda_rows_v2.restype = ctypes.c_int
        library.planar_cuda_correct.argtypes = (ptr, ctypes.c_int, ctypes.c_int, ptr, ctypes.c_int)
        library.planar_cuda_correct.restype = ctypes.c_int
        library.planar_cuda_boundary_pointers.argtypes = (ptr, ctypes.POINTER(ptr), ctypes.POINTER(ptr))
        library.planar_cuda_boundary_pointers.restype = ctypes.c_int
    if hasattr(library, "planar_cuda_refresh"):
        library.planar_cuda_refresh.argtypes = (ptr,) * 8 + (ctypes.c_double,) * 4
        library.planar_cuda_refresh.restype = ctypes.c_int
    return library


_CUDA_NEAR_DTYPE = np.dtype({
    "names": ("index", "argument", "single", "double"),
    "formats": (np.int32, np.complex128, np.complex128, np.complex128),
    "offsets": (0, 8, 24, 40),
    "itemsize": 56,
})
_CUDA_CORRECTION_DTYPE = np.dtype({
    "names": ("index", "single", "double"),
    "formats": (np.int32, np.complex128, np.complex128),
    "offsets": (0, 8, 24),
    "itemsize": 40,
})


def _check_cuda_assembly(library, status):
    if status:
        message = library.planar_cuda_error(status).decode()
        raise RuntimeError(f"Planar BEM CUDA assembly failed: {message}")


def _record_time(timings, name, elapsed):
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + elapsed


def _repeated_k0(argument):
    # Straight panel groups repeat identical radii. Keep AMOS calls on the
    # negative-real axis separate, where opposite signed zeros can distinguish
    # sides of the branch cut. Elsewhere identical arguments share one value.
    shape = argument.shape
    argument = argument.ravel()
    on_cut = (argument.real < 0) & (argument.imag == 0)
    repeated = ~on_cut & np.isfinite(argument)
    distinct, inverse = np.unique(argument[repeated], return_inverse=True)
    single = np.empty_like(argument)
    single[repeated] = kv(0, distinct)[inverse]
    single[~repeated] = kv(0, argument[~repeated])
    return single.reshape(shape)


def _amos_corrections(records):
    start = perf_counter()
    values = np.zeros(records.size, dtype=_CUDA_CORRECTION_DTYPE)
    values["index"] = records["index"]
    values["single"] = records["single"] * _repeated_k0(records["argument"])
    has_double = records["double"] != 0
    values["double"][has_double] = records["double"][has_double] * kv(1, records["argument"][has_double])
    return values, perf_counter() - start


def _cuda_regular(library, device, wave, targets, geometry, panels, double_columns, correction_pairs,
                  *, single=None, double=None, scale_coupling=None, executor=None, amos_queue=1, tile_rows=128,
                  cached=None, timings=None):
    """Assemble row tiles on the GPU and complete the AMOS samples on the host."""
    rows = len(targets[0])
    tile_rows = min(tile_rows, rows)
    targets = tuple(np.ascontiguousarray(item) for item in targets)
    geometry = tuple(np.ascontiguousarray(item) for item in geometry)
    has_v2 = hasattr(library, "planar_cuda_prepare")
    resident = scale_coupling is not None
    if resident and not has_v2:
        raise RuntimeError("Device-resident planar BEM requires a rebuilt CUDA assembly library")
    start = perf_counter()
    can_cache = resident and cached is not None and hasattr(library, "planar_cuda_refresh")
    storage = cached.pop(timings is not None, None) if can_cache else None
    if storage is None:
        handle = ctypes.c_void_p()
        _check_cuda_assembly(library, library.planar_cuda_create(
            device, panels, double_columns, tile_rows, len(geometry[-1]),
            *(item.ctypes.data for item in geometry), ctypes.byref(handle),
        ))
        storage = _CudaWorkspace(library, handle) if resident else None
        new_workspace = True
    else:
        handle = storage.handle
        new_workspace = False
    owner = _CudaBoundary(storage, cached if can_cache else None, timings is not None) if resident else None
    try:
        if has_v2 and new_workspace:
            pair_rows, pair_columns = correction_pairs
            offsets = np.r_[0, np.cumsum(np.bincount(pair_rows, minlength=rows))].astype(np.int32)
            columns = pair_columns.astype(np.int32, copy=False)
            coupling = scale_coupling if resident else 0j
            initial_capacity = min(tile_rows * panels * len(geometry[-1]), max(16384, tile_rows * panels // 6))
            _check_cuda_assembly(library, library.planar_cuda_prepare(
                handle, rows, len(columns), offsets.ctypes.data, columns.ctypes.data, resident,
                -wave.real, -wave.imag, coupling.real, coupling.imag, timings is not None, initial_capacity,
            ))
        if can_cache:
            _check_cuda_assembly(library, library.planar_cuda_refresh(
                handle, *(item.ctypes.data for item in geometry[:-1]), *(item.ctypes.data for item in targets),
                -wave.real, -wave.imag, scale_coupling.real, scale_coupling.imag,
            ))
        _record_time(timings, "bem_cuda_setup_s", perf_counter() - start)
        queued = deque()
        # CUDA calls and updates stay on this thread. AMOS workers take a copy
        # so the next CUDA tile can reuse the native staging buffer.
        for first in range(0, rows, tile_rows):
            tile = slice(first, first + tile_rows)
            count = min(rows - first, tile_rows)
            near, size = ctypes.c_void_p(), ctypes.c_int()
            kernel = ctypes.c_double()
            outputs = (0, 0) if resident else (single[tile].ctypes.data, double[tile].ctypes.data)
            payload = (*(item[tile].ctypes.data for item in targets), *outputs, ctypes.byref(near), ctypes.byref(size))
            start = perf_counter()
            if has_v2:
                status = library.planar_cuda_rows_v2(handle, first, count, wave.real, wave.imag,
                                                     *payload, ctypes.byref(kernel))
            else:
                status = library.planar_cuda_rows(handle, count, wave.real, wave.imag, *payload)
            _check_cuda_assembly(library, status)
            _record_time(timings, "bem_cuda_kernel_s", kernel.value)
            _record_time(timings, "bem_cuda_tile_s", perf_counter() - start)
            if size.value:
                records = np.frombuffer((ctypes.c_char * (size.value * _CUDA_NEAR_DTYPE.itemsize))
                                        .from_address(near.value), dtype=_CUDA_NEAR_DTYPE)
                if executor is not None:
                    queued.append((first, executor.submit(_amos_corrections, records.copy())))
                    if len(queued) > amos_queue:
                        owner.complete_amos(*queued.popleft(), timings)
                elif resident:
                    values, elapsed = _amos_corrections(records)
                    _record_time(timings, "bem_cuda_amos_s", elapsed)
                    owner.correct(first, values, replace=False, timings=timings)
                else:
                    start = perf_counter()
                    _complete_regular(records["index"], records["argument"], records["single"], records["double"],
                                      single[tile], double[tile])
                    _record_time(timings, "bem_cuda_amos_s", perf_counter() - start)
        while queued:
            owner.complete_amos(*queued.popleft(), timings)
        if resident:
            return owner
    except BaseException:
        if resident:
            owner.close()
        raise
    finally:
        if not resident:
            library.planar_cuda_destroy(handle)


class _CudaBoundary:
    def __init__(self, storage, cache, profile):
        self.storage = storage
        self.library, self.handle = storage.library, storage.handle
        self.cache = cache
        self.profile = profile
        self.boundary = 0
        self.coupling = 0

    def correct(self, first, values, *, replace, timings):
        start = perf_counter()
        _check_cuda_assembly(self.library, self.library.planar_cuda_correct(
            self.handle, first, len(values), values.ctypes.data, replace,
        ))
        _record_time(timings, "bem_cuda_correction_upload_s", perf_counter() - start)

    def complete_amos(self, first, pending, timings):
        start = perf_counter()
        values, elapsed = pending.result()
        _record_time(timings, "bem_cuda_amos_wait_s", perf_counter() - start)
        _record_time(timings, "bem_cuda_amos_s", elapsed)
        self.correct(first, values, replace=False, timings=timings)

    def finish(self):
        boundary, coupling = ctypes.c_void_p(), ctypes.c_void_p()
        _check_cuda_assembly(self.library, self.library.planar_cuda_boundary_pointers(
            self.handle, ctypes.byref(boundary), ctypes.byref(coupling),
        ))
        self.boundary, self.coupling = boundary.value, coupling.value

    def close(self):
        if self.handle:
            if self.cache is not None and self.boundary:
                self.cache.setdefault(self.profile, self.storage)
            else:
                self.storage.close()
            self.storage = None
            self.handle = None
            self.boundary = self.coupling = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _CudaWorkspace:
    def __init__(self, library, handle):
        self.library = library
        self.handle = handle
        self.process = os.getpid()

    def close(self):
        if self.handle and self.process == os.getpid():
            self.library.planar_cuda_destroy(self.handle)
        self.handle = None

    __del__ = close


class _CorrectionAxis:
    """Frequency-independent sample map for one axis of the PML contour."""

    def __init__(self, target, source, active_target, active_source, uses_derivative, stretch_axis):
        self.stretch_axis = stretch_axis
        self.delta = source - target
        self.dynamic = (active_source | active_target) & (target != source)
        self.dynamic_derivative = active_source & uses_derivative
        # Equal coordinates give zero separation even in the PML. For other
        # stretched coordinates only reuse differences of identical pairs.
        dynamic_pairs = np.column_stack((target[self.dynamic], source[self.dynamic]))
        pairs, inverse = np.unique(dynamic_pairs, axis=0, return_inverse=True)
        self.pair_index = inverse.ravel()
        sampled = np.r_[pairs.ravel(), source[self.dynamic_derivative]]
        self.coordinates, inverse = np.unique(sampled, return_inverse=True)
        self.pair_points = inverse[:pairs.size].reshape(pairs.shape)
        self.derivative_points = inverse[pairs.size:]

    def radius_keys(self):
        keys = np.abs(self.delta).view(np.int64)
        keys[self.dynamic] = -(1 + self.pair_index)
        return keys.ravel()

    def evaluate(self, s):
        delta = self.delta.astype(np.complex128)
        derivative = np.ones_like(delta)
        if self.coordinates.size:
            coordinates, sample_derivative = self.stretch_axis(self.coordinates, s)
            differences = coordinates[self.pair_points[:, 1]] - coordinates[self.pair_points[:, 0]]
            delta[self.dynamic] = differences[self.pair_index]
            derivative[self.dynamic_derivative] = sample_derivative[self.derivative_points]
        return delta, derivative


class _BoundaryQuadrature:
    """Reuse identical Bessel arguments for fixed near and diagonal quadrature."""

    def __init__(self, fluid, rows, columns, nodes, weights, *, diagonal=False):
        self.rows = rows
        self.columns = columns
        self.weights = weights
        self.tangent = fluid.vectors[columns]
        self.orientation = fluid.orientation
        self.shape = (len(rows), nodes.size)
        self.stretch = stretch = fluid.stretch
        samples = fluid.starts[columns, None, :] + self.tangent[:, None, :] * nodes[None, :, None]
        centres = np.broadcast_to(fluid.centres[rows, None, :], samples.shape)
        x_source, z_source = samples[..., 0], samples[..., 1]
        x_centre, z_centre = centres[..., 0], centres[..., 1]

        def x_active(value):
            return (stretch.pml_x > 0) & (stretch.damping_x > 0) & ((value < stretch.xmin) | (value > stretch.xmax))

        def z_active(value):
            return (stretch.pml_z > 0) & (stretch.damping_z > 0) & (stretch.z_direction * (value - stretch.zlimit) > 0)

        self.x = _CorrectionAxis(x_centre, x_source, x_active(x_centre), x_active(x_source),
                                 self.tangent[:, 0, None] != 0, stretch.horizontal)
        self.z = _CorrectionAxis(z_centre, z_source, z_active(z_centre), z_active(z_source),
                                 self.tangent[:, 1, None] != 0, stretch.vertical)
        keys = np.column_stack((self.x.radius_keys(), self.z.radius_keys()))
        physical = (keys >= 0).all(axis=1)
        physical_x = self.x.delta.ravel()[physical]
        physical_z = self.z.delta.ravel()[physical]
        # Positive real radii use one separate key plane, so repeated radii
        # with different fixed X/Z decompositions can share an AMOS call too.
        keys[physical, 0] = np.sqrt(physical_x**2 + physical_z**2).view(np.int64)
        keys[physical, 1] = np.iinfo(np.int64).min
        _, self.unique, self.expand = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        self.expand = self.expand.reshape(self.shape)
        self.log_nodes = np.log(abs(nodes - 0.5)) if diagonal else None
        if diagonal:
            self.midpoints = fluid.starts[columns] + 0.5 * self.tangent

    def evaluate(self, s, wave, double_columns):
        dx, derivative_x = self.x.evaluate(s)
        dz, derivative_z = self.z.evaluate(s)
        tx = derivative_x * self.tangent[:, 0, None]
        tz = derivative_z * self.tangent[:, 1, None]
        separation_x, separation_z = dx.ravel()[self.unique], dz.ravel()[self.unique]
        radius = np.sqrt(separation_x**2 + separation_z**2)
        argument = wave * radius
        integrand = kv(0, argument)[self.expand] * np.sqrt(tx**2 + tz**2)
        analytic_log = 0.0
        if self.log_nodes is not None:
            _, vx = self.stretch.horizontal(self.midpoints[:, 0], s)
            _, vz = self.stretch.vertical(self.midpoints[:, 1], s)
            metric = np.sqrt((vx * self.tangent[:, 0])**2 + (vz * self.tangent[:, 1])**2)
            integrand += metric[:, None] * self.log_nodes
            analytic_log = (1.0 + np.log(2.0)) * metric
        single = (np.sum(integrand * self.weights, axis=1) + analytic_log) / (2.0 * np.pi)
        keep = self.columns < double_columns
        dot = self.orientation * (dz[keep] * tx[keep] - dx[keep] * tz[keep])
        nonzero = dot != 0
        lookup = self.expand[keep][nonzero]
        unique, expand = np.unique(lookup, return_inverse=True)
        kernel = np.zeros_like(dot)
        kernel[nonzero] = -wave * kv(1, argument[unique])[expand] * dot[nonzero] / radius[lookup]
        double = np.sum(kernel * self.weights, axis=1) / (2.0 * np.pi)
        return self.rows, self.columns, single, keep, double


class PlanarFluid:
    """Boundary starts on the ice face, runs around the PML, then closes.

    With reference_surface_z=(left, right), the moving side walls keep the
    number of panels of the reference geometry and meet the same PML joins.
    """

    def __init__(self, x_vertices, surface_z, z_outer, panel_size, stretch, sound_speed, *, surface="top", native=True,
                 assembly_gpu_library=None, gpu_device=0, reference_surface_z=None):
        x = np.asarray(x_vertices, dtype=float)
        z = np.asarray(surface_z, dtype=float)
        if (x.ndim != 1 or x.size < 2 or z.shape != x.shape or not np.all(np.diff(x) > 0)
                or not np.isfinite(x).all() or not np.isfinite(z).all()
                or not np.isfinite([z_outer, panel_size, sound_speed]).all()
                or panel_size <= 0 or sound_speed <= 0
                or surface not in {"top", "bottom"}
                or (surface == "top" and z_outer >= z.min())
                or (surface == "bottom" and z_outer <= z.max())):
            raise ValueError("Invalid Cartesian acoustic boundary or panel size")

        if reference_surface_z is None:
            right = split_edges(z[-1], z_outer, panel_size, stretch.zlimit)
            left = split_edges(z_outer, z[0], panel_size, stretch.zlimit)
        else:
            reference = np.asarray(reference_surface_z, dtype=float)
            if reference.shape != (2,) or not np.isfinite(reference).all():
                raise ValueError("Reference interface endpoints must be two finite z coordinates (left, right)")
            right = _fixed_wall_edges(z[-1], z_outer, stretch.zlimit, reference[1], panel_size)
            left = _fixed_wall_edges(z[0], z_outer, stretch.zlimit, reference[0], panel_size, reverse=True)
        far = split_edges(x[-1], x[0], panel_size, stretch.xmin, stretch.xmax)
        self.vertices = np.column_stack((
            np.r_[x, np.full(right.size - 1, x[-1]), far[1:], np.full(left.size - 1, x[0])],
            np.r_[z, right[1:], np.full(far.size - 1, z_outer), left[1:]],
        ))
        self.starts = self.vertices[:-1]
        self.vectors = np.diff(self.vertices, axis=0)
        self.centres = self.starts + 0.5 * self.vectors
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self._correction_radius2 = (2.0 * self.lengths)**2
        self._correction_tree = cKDTree(self.centres)
        self._boundary_pairs = self._correction_pairs(self.centres)
        self.surface_count = x.size - 1
        self.panel_count = self.centres.shape[0]
        self.stretch = stretch
        self.sound_speed = sound_speed
        self.orientation = 1.0 if surface == "top" else -1.0
        self._fixed_corrections = None
        self._assembly_cache = None
        self.native = native
        self.assembly_gpu_library = Path(assembly_gpu_library).resolve() if assembly_gpu_library is not None else None
        self.gpu_device = gpu_device
        self.assembly_threads = int(os.environ.get("PLANAR_ASSEMBLY_THREADS", "1"))
        if self.assembly_threads < 1:
            raise ValueError("PLANAR_ASSEMBLY_THREADS must be a positive integer")
        self.assembly_tile_rows = int(os.environ.get("PLANAR_BEM_TILE_ROWS", "512"))
        if self.assembly_tile_rows < 1:
            raise ValueError("PLANAR_BEM_TILE_ROWS must be a positive integer")

    def _correction_pairs(self, targets):
        radius = np.nextafter(np.sqrt(np.max(self._correction_radius2)), np.inf)
        neighbours = self._correction_tree.query_ball_point(targets, radius, return_sorted=True)
        rows = np.repeat(np.arange(len(targets)), [len(columns) for columns in neighbours])
        columns = np.fromiter((column for group in neighbours for column in group), dtype=np.intp, count=rows.size)
        # The tree supplies candidates; the original strict per-panel metric
        # determines exactly which four-point entries will be replaced.
        within = np.sum((targets[rows] - self.centres[columns])**2, axis=1) < self._correction_radius2[columns]
        return rows[within], columns[within]

    def _transform(self, points, s):
        x, _ = self.stretch.horizontal(points[..., 0], s)
        z, _ = self.stretch.vertical(points[..., 1], s)
        return x, z

    def _geometry(self, columns, nodes, s):
        start = self.starts[columns]
        tangent = self.vectors[columns]
        px = start[:, 0, None] + tangent[:, 0, None] * nodes
        pz = start[:, 1, None] + tangent[:, 1, None] * nodes
        x, dx = self.stretch.horizontal(px, s)
        z, dz = self.stretch.vertical(pz, s)
        tx = dx * tangent[:, 0, None]
        tz = dz * tangent[:, 1, None]
        return x, z, np.sqrt(tx**2 + tz**2), -self.orientation * tz, self.orientation * tx

    def _regular(self, targets, values, s, double_columns, correction_pairs, *, timings=None, use_cuda=True):
        """Integrate all columns in row blocks and only evaluate K1 where needed."""
        x_target, z_target = self._transform(targets, s)
        x, z, ds, nx, nz = self._geometry(slice(None), values[0], s)
        weights = values[1] / (2.0 * np.pi)
        single = np.zeros((len(targets), self.panel_count), dtype=np.complex128)
        double = np.zeros((len(targets), double_columns), dtype=np.complex128)
        wave = s / self.sound_speed
        if self.assembly_gpu_library is not None and use_cuda:
            _cuda_regular(_load_cuda_assembly(self.assembly_gpu_library), self.gpu_device, wave, (x_target, z_target),
                          (x, z, ds, nx, nz, weights), self.panel_count, double_columns, correction_pairs,
                          single=single, double=double, timings=timings)
            return single, double

        library = build_planar_library() if self.native and self.panel_count >= 250 else None
        def assemble_rows(first):
            rows = slice(first, first + 64)
            if library is not None:
                arrays = (x_target[rows], z_target[rows], x, z, ds, nx, nz, weights)
                _native_block(library, wave, arrays, single[rows], double[rows])
                return
            for ix, weight in enumerate(weights):
                dx = x[None, :, ix] - x_target[rows, None]
                dz = z[None, :, ix] - z_target[rows, None]
                radius = np.sqrt(dx**2 + dz**2)
                single[rows] += weight * kv(0, wave * radius) * ds[None, :, ix]
                if double_columns:
                    k_radius = radius[:, :double_columns]
                    normal_dot = (dx[:, :double_columns] * nx[None, :double_columns, ix]
                                  + dz[:, :double_columns] * nz[None, :double_columns, ix])
                    double[rows] -= weight * wave * kv(1, wave * k_radius) * normal_dot / k_radius

        starts = range(0, len(targets), 64)
        threads = min(self.assembly_threads, len(starts))
        if threads > 1:
            # Each row block writes to disjoint output and owns its near-field
            # temporaries. Load the native library before entering the pool.
            with ThreadPoolExecutor(max_workers=threads) as pool:
                for _ in pool.map(assemble_rows, starts):
                    pass
        else:
            for first in starts:
                assemble_rows(first)
        return single, double

    def _accurate_pairs(self, targets, rows, columns, s, nodes, weights, double_columns):
        x, z, ds, nx, nz = self._geometry(columns, nodes, s)
        target_rows, index = np.unique(rows, return_inverse=True)
        x_target, z_target = (part[index] for part in self._transform(targets[target_rows], s))
        dx = x - x_target[:, None]
        dz = z - z_target[:, None]
        radius = np.sqrt(dx**2 + dz**2)
        wave = s / self.sound_speed
        single = np.sum(kv(0, wave * radius) * ds * weights, axis=-1) / (2.0 * np.pi)
        keep = columns < double_columns
        normal_dot = dx[keep] * nx[keep] + dz[keep] * nz[keep]
        k_radius = radius[keep]
        # Horizontal/vertical source fragments with a zero normal projection
        # contribute exactly zero. AMOS is only needed at the remaining nodes.
        nonzero = normal_dot != 0
        kernel = np.zeros_like(k_radius)
        kernel[nonzero] = -wave * kv(1, wave * k_radius[nonzero]) * normal_dot[nonzero] / k_radius[nonzero]
        double = np.sum(kernel * weights[keep], axis=-1) / (2.0 * np.pi)
        return single, keep, double

    def _receiver_pairs(self, targets, rows, columns):
        # Resolve the narrow near-singularity around the closest point to the
        # receiver with t=t0±d*sinh(u). This covers endpoint targets as well.
        delta = targets[rows] - self.starts[columns]
        tangent = self.vectors[columns]
        lengths = self.lengths[columns]
        projection = np.clip(np.sum(delta * tangent, axis=1) / lengths**2, 0.0, 1.0)
        minimum = np.linalg.norm(delta - tangent * projection[:, None], axis=1) / lengths
        if np.any(minimum <= np.finfo(float).eps):
            raise ValueError("Receiver lies on an acoustic boundary panel")
        points, weights = _quadrature(48)
        limit = np.arcsinh(np.column_stack((projection, 1.0 - projection)) / minimum[:, None])
        u = limit[..., None] * points
        nodes = projection[:, None, None] + np.array([-1.0, 1.0])[None, :, None] * minimum[:, None, None] * np.sinh(u)
        jacobian = minimum[:, None, None] * np.cosh(u) * limit[..., None]
        return nodes.reshape(len(rows), -1), (weights * jacobian).reshape(len(rows), -1)

    def _accurate_corrections(self, s, targets, double_columns, correction_pairs, boundary):
        if boundary:
            if self._fixed_corrections is None:
                rows, columns = correction_pairs
                near = rows != columns
                u, w = _quadrature(24)
                diagonal = np.arange(self.panel_count)
                # t = (1 ± u²)/2 regularizes the logarithmic diagonal singularity.
                nodes = np.r_[0.5 * (1.0 - u**2), 0.5 * (1.0 + u**2)]
                self._fixed_corrections = (
                    _BoundaryQuadrature(self, rows[near], columns[near], *_quadrature(12)),
                    _BoundaryQuadrature(self, diagonal, diagonal, nodes, np.tile(u * w, 2), diagonal=True),
                )
            wave = s / self.sound_speed
            for quadrature in self._fixed_corrections:
                if quadrature.rows.size:
                    yield quadrature.evaluate(s, wave, double_columns)
            return
        rows, columns = correction_pairs
        if rows.size:
            quadrature = self._receiver_pairs(targets, rows, columns)
            values, keep, k_values = self._accurate_pairs(targets, rows, columns, s, *quadrature, double_columns)
            yield rows, columns, values, keep, k_values

    def _assemble(self, s, targets, double_columns, *, boundary=False, timings=None, use_cuda=True):
        correction_pairs = self._boundary_pairs if boundary else self._correction_pairs(targets)
        single, double = self._regular(targets, _quadrature(4), s, double_columns, correction_pairs,
                                       timings=timings, use_cuda=use_cuda)
        start = perf_counter()
        for rows, columns, values, keep, k_values in self._accurate_corrections(
            s, targets, double_columns, correction_pairs, boundary,
        ):
            single[rows, columns] = values
            double[rows[keep], columns[keep]] = k_values
        if not np.isfinite(single).all() or not np.isfinite(double).all():
            raise FloatingPointError(f"Nonfinite 2D layer potential at Laplace point {s}")
        _record_time(timings, "bem_accurate_corrections_s", perf_counter() - start)
        return single, double

    def assemble(self, s, receiver=None, *, double_columns=None, timings=None):
        """Return boundary integral kernels in (I/2 + K)p - S q = G."""
        s = complex(s)
        if not np.isfinite(s) or s.real <= 0:
            raise ValueError("Laplace point must lie in the open right half-plane")
        double_columns = self.panel_count if double_columns is None else int(double_columns)
        if not 0 <= double_columns <= self.panel_count:
            raise ValueError("Invalid number of double-layer columns")
        target, scalar_receiver = self._receiver_targets(receiver)
        single, double = self._assemble(s, self.centres, double_columns, boundary=True, timings=timings)
        result = {"S": single, "K": double}
        if target is not None:
            sx, kx = self._assemble(s, target, double_columns, timings=timings)
            if scalar_receiver:
                sx, kx = sx[0], kx[0]
            result.update(S_receiver=sx, K_receiver=kx)
        return result

    def assemble_device(self, s, receiver, *, density, timings=None):
        """Assemble one GPU-resident water system and CPU receiver rows in input order."""
        if self.assembly_gpu_library is None:
            raise ValueError("Device-resident assembly requires the CUDA BEM library")
        s = complex(s)
        if not np.isfinite(s) or s.real <= 0 or not np.isfinite(density) or density <= 0:
            raise ValueError("Invalid Laplace point or acoustic density")
        process = os.getpid()
        if self._assembly_cache is None or self._assembly_cache[0] != process:
            self._assembly_cache = (process, {})
        target, scalar_receiver = self._receiver_targets(receiver)
        pool = (ThreadPoolExecutor(max_workers=self.assembly_threads) if self.assembly_threads > 1
                else nullcontext())
        owner = None
        try:
            with pool as executor:
                pending = executor.submit(self._device_corrections, s, target) if executor is not None else None
                points, weights = _quadrature(4)
                geometry = (*self._geometry(slice(None), points, s), weights / (2.0 * np.pi))
                owner = _cuda_regular(
                    _load_cuda_assembly(self.assembly_gpu_library), self.gpu_device, s / self.sound_speed,
                    self._transform(self.centres, s), geometry, self.panel_count, self.surface_count, self._boundary_pairs,
                    scale_coupling=density * s**2, executor=executor, amos_queue=self.assembly_threads,
                    tile_rows=self.assembly_tile_rows, cached=self._assembly_cache[1], timings=timings,
                )
                if pending is not None:
                    start = perf_counter()
                    corrections, sx, kx, elapsed, receiver_elapsed = pending.result()
                    _record_time(timings, "bem_device_corrections_wait_s", perf_counter() - start)
                else:
                    corrections, sx, kx, elapsed, receiver_elapsed = self._device_corrections(s, target)
            for block in corrections:
                owner.correct(0, block, replace=True, timings=timings)
            _record_time(timings, "bem_accurate_corrections_s", elapsed)
            _record_time(timings, "bem_receiver_cpu_s", receiver_elapsed)
            owner.finish()
            if scalar_receiver:
                sx, kx = sx[0], kx[0]
            owner.single_receiver, owner.double_receiver = sx, kx
            return owner
        except BaseException:
            if owner is not None:
                owner.close()
            raise

    def _device_corrections(self, s, target):
        start = perf_counter()
        corrections = []
        for rows, columns, values, keep, k_values in self._accurate_corrections(
            s, self.centres, self.surface_count, self._boundary_pairs, True,
        ):
            if not np.isfinite(values).all() or not np.isfinite(k_values).all():
                raise FloatingPointError(f"Nonfinite 2D layer potential at Laplace point {s}")
            block = np.zeros(len(rows), dtype=_CUDA_CORRECTION_DTYPE)
            block["index"] = rows * self.panel_count + columns
            block["single"] = values
            block["double"][keep] = k_values
            corrections.append(block)
        elapsed = perf_counter() - start
        start = perf_counter()
        if target is None:
            sx = kx = None
        else:
            sx, kx = self._assemble(s, target, self.surface_count, use_cuda=False)
        return corrections, sx, kx, elapsed, perf_counter() - start

    @staticmethod
    def _receiver_targets(receiver):
        if receiver is None:
            return None, False
        values = np.asarray(receiver, dtype=float)
        scalar = values.shape == (2,)
        targets = np.atleast_2d(values)
        if targets.ndim != 2 or not len(targets) or targets.shape[1] != 2 or not np.isfinite(targets).all():
            raise ValueError("Receiver coordinates must be one (x, z) point or a finite nonempty (n, 2) array")
        return targets, scalar

    def incident(self, s, source, receiver=None):
        """Point-source pressure on all panels and optionally at one or more receivers."""
        target, scalar_receiver = self._receiver_targets(receiver)
        points = self.centres if target is None else np.vstack((self.centres, target))
        tx, tz = self._transform(points, s)
        sx, sz = self._transform(np.asarray(source, dtype=float).reshape(1, 2), s)
        distance = np.sqrt((tx - sx[0])**2 + (tz - sz[0])**2)
        if np.any(distance == 0):
            raise ValueError("Source coincides with a boundary or receiver point")
        value = free_green(s, distance, self.sound_speed)
        if target is None:
            return value, None
        observed = value[self.panel_count:]
        return value[:self.panel_count], observed[0] if scalar_receiver else observed
