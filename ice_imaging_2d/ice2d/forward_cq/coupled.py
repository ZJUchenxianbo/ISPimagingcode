#!/usr/bin/env python3
"""Receiver waveform from planar water/air CQ-BEM coupled to elastic ice FEM."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from time import perf_counter

BLAS_THREADS = os.environ.get("PLANAR_BLAS_THREADS", "1")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, BLAS_THREADS)

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import splu

from ..geometry import expression
from .water import PlanarStretch, split_edges
from .receiver import CauchyFluid as PlanarFluid
from .ice import PlanarIce
from .trace import TraceFactor, TraceFactorError


RESIDUAL_TOLERANCE = 2e-8


@contextmanager
def phase(timings, name):
    if timings is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + (perf_counter() - start)


class ProfiledIce(PlanarIce):
    timings = None

    def dynamic_matrix(self, s):
        with phase(self.timings, "ice_dynamic_matrix_s"):
            return super().dynamic_matrix(s)


@dataclass(frozen=True)
class Case:
    width: float
    depth: float
    pml_x: float
    pml_z: float
    f1: str
    f2: str
    source: tuple
    receiver: tuple
    frequency: float
    air: str
    mesh_scale: float = 1.0
    ice_elements_per_wavelength: float | None = None
    water_panel_m: float | None = None
    water_speed: float = 1500.0
    water_density: float = 1025.0
    ice_density: float = 920.0
    ice_p_speed: float = 3600.0
    ice_s_speed: float = 1800.0
    air_speed: float = 343.0
    air_density: float = 1.225
    native: bool = False
    cauchy: bool = True


@dataclass(frozen=True)
class Geometry:
    edges: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    air_edges: np.ndarray
    air_top: np.ndarray
    spacing: float
    air_spacing: float
    ice_spacing: float | None
    z_layers: int
    air_ceiling: float
    low: float
    high: float


def subdivide(edges, segment_lengths, spacing):
    counts = np.maximum(1, np.ceil(segment_lengths / spacing).astype(int))
    return np.concatenate([np.concatenate([np.linspace(a, b, n, endpoint=False)
                                           for a, b, n in zip(edges[:-1], edges[1:], counts)]), edges[-1:]])


def make_geometry(case, design_frequency, *, interface_profiles=None, reference_geometry=None,
                  exterior_profiles=None):
    low, high = -case.width / 2, case.width / 2
    spacing = (case.water_panel_m if case.water_panel_m is not None
               else case.mesh_scale * case.water_speed / (8 * design_frequency))
    air_spacing = case.mesh_scale * case.air_speed / (8 * design_frequency)
    edges = split_edges(low - case.pml_x, high + case.pml_x, spacing, low, high)

    def extend(profile, exterior, x):
        clamped = np.clip(x, low, high)
        if isinstance(profile, str):
            values = expression(profile, clamped)
        else:
            values = np.broadcast_to(np.asarray(profile(clamped), dtype=float), clamped.shape)
        if exterior is not None:
            outside = x != clamped
            values = values.copy()
            values[outside] = exterior(x[outside])
        if not np.isfinite(values).all():
            raise ValueError("Interface profiles must give finite heights")
        return values

    top_profile, bottom_profile = interface_profiles or (case.f1, case.f2)
    exterior_top, exterior_bottom = exterior_profiles or (None, None)
    if reference_geometry is not None:
        if case.air != "zero-traction" or (low, high, spacing) != (
                reference_geometry.low, reference_geometry.high, reference_geometry.spacing):
            raise ValueError("Fixed mesh topology requires a matching zero-traction reference mesh")
        edges = reference_geometry.edges
    bottom, top = extend(bottom_profile, exterior_bottom, edges), extend(top_profile, exterior_top, edges)
    if reference_geometry is None:
        lengths = np.hypot(np.diff(edges), np.maximum(abs(np.diff(bottom)), abs(np.diff(top))))
        if np.any(lengths > spacing * (1 + 1e-12)):
            edges = subdivide(edges, lengths / (1 + 1e-12), spacing)
            bottom, top = extend(bottom_profile, exterior_bottom, edges), extend(top_profile, exterior_top, edges)
    if np.any(top <= bottom) or np.any(bottom <= -case.depth):
        raise ValueError("Require -depth < f2(x) < f1(x) on every sampled interface point")
    for name, point in (("source", case.source), ("receiver", case.receiver)):
        x, z = point
        if not (low < x < high and -case.depth < z < np.interp(x, edges, bottom)):
            raise ValueError(f"The {name} must lie inside the physical water domain")
    if tuple(case.source) == tuple(case.receiver):
        raise ValueError("Source and receiver must be at distinct points")
    if case.air != "zero-traction":
        air_edges = subdivide(edges, np.hypot(np.diff(edges), np.diff(top)), air_spacing)
        air_top = np.interp(air_edges, edges, top)
    else:
        air_edges, air_top = np.empty(0), np.empty(0)
    ice_resolution = case.ice_elements_per_wavelength
    ice_spacing = case.mesh_scale * case.ice_s_speed / ((ice_resolution or 3.0) * design_frequency)
    layers = max(2, math.ceil(np.max(top - bottom) / ice_spacing))
    if reference_geometry is not None:
        layers = reference_geometry.z_layers
    return Geometry(edges, bottom, top, air_edges, air_top, spacing, air_spacing,
                    ice_spacing if ice_resolution is not None else None, layers,
                    float(top.max()), low, high)


class PlanarTransfer:
    """Shares the water assembly and ice factorization across an air A/B run."""

    def __init__(self, case, design_frequency, *, mesh_only=False, timings=None,
                 zero_solver="direct", mumps_library=None, water_preconditioner="dense",
                 cudss_library=None, gpu_trace_library=None, gpu_device=0, water_gpu_library=None,
                 water_gmres_restart=60, bem_gpu_library=None, gpu_trace_full_ice=False,
                 gpu_panel_schur_library=None, gpu_resident_bem=False, gpu_water_workspace=False,
                 gpu_ice_residual=False, gpu_ice_assembly=False, gpu_water_refinement=False,
                 gpu_water_fast_refinement=False, gpu_ice_nd_levels=None, gpu_ice_normal=False,
                 interface_profiles=None, reference_geometry=None, exterior_profiles=None,
                 ice_assembly_layout=None, cache_ice_assembly_layout=False):
        if zero_solver != "direct" and case.air != "zero-traction":
            raise ValueError("The condensed water solver requires --air zero-traction")
        self.case = case
        self.condensation = None
        self.full_ice = None
        self.mumps_library = mumps_library
        self.cudss_library = cudss_library
        self.gpu_trace_library = gpu_trace_library
        self.gpu_panel_schur_library = gpu_panel_schur_library
        self.gpu_ice_residual = gpu_ice_residual
        self.gpu_ice_assembly = gpu_ice_assembly
        self.gpu_ice_nd_levels = gpu_ice_nd_levels
        self.gpu_ice_normal = gpu_ice_normal
        self.gpu_water_refinement = gpu_water_refinement or gpu_water_fast_refinement
        self.gpu_water_fast_refinement = gpu_water_fast_refinement
        self.gpu_device = gpu_device
        self.water_gpu_library = water_gpu_library
        self.ice_factor = None
        self.gpu_resident_bem = gpu_resident_bem
        self.gpu_water_workspace = gpu_water_workspace or gpu_resident_bem or self.gpu_water_refinement
        self.water_direct = None
        self.water_preconditioner = water_preconditioner
        self.water_gmres_restart = water_gmres_restart
        with phase(timings, "mesh_and_fluid_geometry_s"):
            g = self.geometry = make_geometry(case, design_frequency, interface_profiles=interface_profiles,
                                              reference_geometry=reference_geometry, exterior_profiles=exterior_profiles)
            relative_reflection = 1e-3
            damping = -1.5 * np.log(relative_reflection)
            lateral = damping * self.case.ice_p_speed / case.pml_x if case.pml_x else 0.0
            alpha = np.pi * case.frequency
            pml_water = PlanarStretch(g.low, g.high, -case.depth, case.pml_x, case.pml_z, lateral,
                                      damping * self.case.water_speed / case.pml_z if case.pml_z else 0.0, alpha)
            assembly_options = ({"assembly_gpu_library": bem_gpu_library, "gpu_device": gpu_device}
                                if bem_gpu_library is not None else {})
            if reference_geometry is not None:
                assembly_options["reference_surface_z"] = reference_geometry.bottom[[0, -1]]
            self.water = PlanarFluid(g.edges, g.bottom, -case.depth - case.pml_z, g.spacing,
                                     pml_water, self.case.water_speed, surface="top", native=case.native, **assembly_options)
            self.water.cauchy = case.cauchy
            self.air = None
            if case.air != "zero-traction":
                # A short physical buffer separates the highest ice crest from vertical PML.
                air_join = g.air_ceiling + max(2 * self.case.air_speed / case.frequency, 2 * g.air_spacing)
                pml_air = PlanarStretch(g.low, g.high, air_join, case.pml_x, case.pml_z, lateral,
                                        damping * self.case.air_speed / case.pml_z if case.pml_z else 0.0, alpha, z_direction=1)
                self.air = PlanarFluid(g.air_edges, g.air_top, air_join + case.pml_z, g.air_spacing,
                                       pml_air, self.case.air_speed, surface="bottom", native=case.native)
        with phase(timings, "ice_static_matrices_s"):
            self.ice = ProfiledIce(g.edges, g.bottom, g.top, stretch=pml_water, x_bounds=(g.low, g.high),
                                   z_layers=g.z_layers, degree=4, panels_per_element=2, clamp_outer=True,
                                     rho=case.ice_density, cp=case.ice_p_speed, cs=case.ice_s_speed,
                                   element_size=g.ice_spacing,
                                   build_matrices=not mesh_only, build_global_matrices=zero_solver == "direct")
        with phase(timings, "ice_trace_coordinates_s"):
            self.trace = None
            if not mesh_only:
                if zero_solver == "direct":
                    self.trace = TraceFactor(self.ice)
                elif gpu_trace_full_ice or gpu_panel_schur_library is not None:
                    from .uncondensed import PlanarUncondensed

                    self.full_ice = PlanarUncondensed(self.ice, native=case.native, layout=ice_assembly_layout,
                                                     cache_layout=cache_ice_assembly_layout)
                    if gpu_ice_assembly:
                        self.full_ice.prepare_device_assembly()
                else:
                    from .condensed import ElementCondensation

                    self.condensation = ElementCondensation(self.ice)
                    if gpu_trace_library is not None:
                        self.condensation.prepare_upper()

    def mesh_summary(self):
        mesh = {"water_panels": self.water.panel_count, "water_ice_panels": self.water.surface_count,
                "air_panels": self.air.panel_count if self.air else 0,
                "air_ice_panels": self.air.surface_count if self.air else 0,
                "ice_elements": self.ice.nr_elements * self.ice.z_layers,
                "ice_displacement_dofs": self.ice.ndof, "ice_vertical_layers": self.ice.z_layers,
                "water_target_panel_m": self.geometry.spacing, "air_target_panel_m": self.geometry.air_spacing}
        if self.geometry.ice_spacing is not None:
            ice = self.ice
            interior = 2 * (ice.degree - 1)**2 * mesh["ice_elements"]
            mesh.update(ice_horizontal_elements=ice.nr_elements, ice_displacement_degree=ice.degree,
                        ice_condensed_dofs=ice.ndof - interior,
                        ice_target_element_m=self.geometry.ice_spacing,
                        ice_max_dx_m=float(np.diff(ice.element_edges).max()),
                        ice_max_dz_m=float(np.max(ice.top - ice.bottom) / ice.z_layers))
        return mesh

    @staticmethod
    def boundary_system(fluid, s, density, receiver=None, *, timings=None, prefix="water", receiver_only=False):
        n = fluid.surface_count
        wave = s / fluid.sound_speed
        with phase(timings, f"{prefix}_bem_s"):
            data = fluid.assemble(s, receiver, double_columns=n)
        with phase(timings, f"{prefix}_boundary_blocks_s"):
            single = data["S"]
            system = np.column_stack((data["K"], -wave * single[:, n:]))
            diagonal = np.arange(n)
            system[diagonal, diagonal] += 0.5
            coupling = (density * s**2) * np.ascontiguousarray(single[:, :n])
        if receiver_only:
            data = {key: data[key] for key in ("S_receiver", "K_receiver")} if receiver is not None else {}
        return system, coupling, wave, data

    def _response(self, s, load, average, timings):
        try:
            # Factor includes dynamic sparse assembly, reported as a separate subphase.
            with phase(timings, "ice_factor_s"):
                factor = self.trace.factor(s, load, average)
            with phase(timings, "ice_panel_response_s"):
                return factor, factor.panel_response(), None
        except TraceFactorError:
            with phase(timings, "ice_factor_s"):
                matrix = self.ice.dynamic_matrix(s)
                factor = splu(matrix)
            with phase(timings, "ice_panel_response_s"):
                response = np.empty((average.shape[0], load.shape[1]), dtype=np.complex128)
                for start in range(0, load.shape[1], 32):
                    sl = slice(start, start + 32)
                    response[:, sl] = average @ factor.solve(load[:, sl].toarray(order="F"))
            return factor, response, matrix

    @staticmethod
    def dense_solve(matrix, rhs, timings=None, prefix="zero"):
        with phase(timings, f"{prefix}_dense_lu_s"):
            factors = lu_factor(matrix, overwrite_a=True, check_finite=False)
        with phase(timings, f"{prefix}_dense_backsolve_s"):
            return lu_solve(factors, rhs, check_finite=False)

    def _factor_condensed_ice(self, s, timings):
        condensed = self.condensation
        with phase(timings, "source_and_ice_trace_maps_s"):
            load, average = self.ice.coupling(s, "bottom")
            if condensed is None:
                reduced_load, reduced_average = load, average
            else:
                retained = condensed.retained_dofs
                reduced_load = load[retained].tocsr()
                if reduced_load.nnz != load.nnz:
                    raise RuntimeError("Ice鈥搘ater traction reached an element-interior degree of freedom")
                reduced_average = average[:, retained].tocsr()
        assembler = condensed if condensed is not None else self.full_ice
        assembly_phase = "ice_element_condensation_s" if condensed is not None else "ice_full_matrix_assembly_s"
        matrix = None
        if not self.gpu_ice_assembly:
            with phase(timings, assembly_phase):
                matrix = assembler.assemble(s)
        with phase(timings, "ice_factor_s"):
            if self.gpu_panel_schur_library is not None:
                from .gpu.panel_schur import GpuPanelSchur

                if self.ice_factor is None:
                    ordering = {"nd_levels": self.gpu_ice_nd_levels} if self.gpu_ice_nd_levels is not None else {}
                    self.ice_factor = GpuPanelSchur(self.gpu_panel_schur_library, device=self.gpu_device, **ordering)
                if self.gpu_ice_assembly:
                    factor = self.ice_factor.factor_assembled(
                        s, reduced_load, reduced_average, assembler=assembler, timings=timings,
                    )
                else:
                    factor = self.ice_factor.factor(
                        s, reduced_load, reduced_average, matrix=matrix,
                        upper_csr=assembler.upper_csr(matrix), timings=timings,
                    )
            elif self.gpu_trace_library is not None:
                from .gpu.trace_factor import GpuTraceFactor

                if self.ice_factor is None:
                    self.ice_factor = GpuTraceFactor(
                        self.ice, device=self.gpu_device, library_path=self.gpu_trace_library,
                    )
                self.ice_factor.timing = timings
                try:
                    factor = self.ice_factor.factor(
                        s, reduced_load, reduced_average, matrix=matrix, upper_csr=assembler.upper_csr(matrix),
                    )
                finally:
                    self.ice_factor.timing = None
            elif self.cudss_library is not None:
                from .gpu.cudss import PlanarCudss

                if self.ice_factor is None:
                    self.ice_factor = PlanarCudss(self.cudss_library, device=self.gpu_device, verbose=True)
                factor = self.ice_factor.factor(matrix)
                if timings is not None:
                    timings["ice_gpu_peak_estimate_bytes"] = factor.info["device_memory_peak_estimate"]
                    timings["ice_gpu_symbolic_reused"] = int(factor.info["reused_analysis"])
            elif self.mumps_library is not None:
                from .mumps import PlanarMumps

                if self.ice_factor is None:
                    self.ice_factor = PlanarMumps(self.mumps_library, ordering="scotch", verbose=True)
                factor = self.ice_factor.factor(matrix)
            else:
                factor = splu(matrix.tocsc(), permc_spec="MMD_AT_PLUS_A")
        del matrix
        if timings is not None:
            timings["ice_condensed_dofs"] = assembler.ndof
        return factor, load, average, reduced_load, reduced_average

    def _factor_resident_system(self, s, receivers, timings):
        """Build the independent ice factor and resident water blocks."""
        def factor_ice():
            with phase(timings, "tangent_ice_factor_s"):
                return self._factor_condensed_ice(s, timings)

        if os.environ.get("PLANAR_OVERLAP_GPU_SETUP", "0") != "1":
            factor = factor_ice()
            with phase(timings, "water_bem_s"):
                resident = self.water.assemble_device(s, receivers, density=self.case.water_density, timings=timings)
            return factor, resident

        water_timing = {} if timings is not None else None

        def assemble_water():
            with phase(water_timing, "water_bem_s"):
                return self.water.assemble_device(s, receivers, density=self.case.water_density, timings=water_timing)

        with phase(timings, "resident_setup_wall_s"), ThreadPoolExecutor(1) as water_executor:
            pending = water_executor.submit(assemble_water)
            try:
                factor = factor_ice()
            except BaseException:
                if not pending.cancel() and pending.exception() is None:
                    pending.result().close()
                raise
            resident = pending.result()
        if timings is not None:
            timings.update(water_timing)
        return factor, resident

    def _evaluate_condensed(self, s, timings, *, receivers=None, sources=None, displacements=None):
        case = self.case
        receiver = case.receiver if receivers is None else receivers
        if sources is None:
            rhs, direct = self.water.incident(s, case.source, receiver)
        else:
            columns = [self.water.incident(s, source, receiver) for source in sources]
            rhs, direct = (np.column_stack(values) for values in zip(*columns))
        nw = self.water.surface_count
        if self.gpu_resident_bem:
            (factor, load, _, reduced_load, reduced_average), resident = self._factor_resident_system(s, receiver, timings)
            incident = rhs[:, None] if sources is None else rhs
            scatter, residuals = [], []
            kw = s / self.water.sound_speed
            with resident:
                for solution, displacement, normal, water_residual in self._solve_resident_shots(
                        factor, incident, reduced_load, reduced_average, timings, resident,
                        keep_displacement=displacements is not None):
                    ice_residual = self._condensed_ice_residual(
                        s, factor, load, displacement, solution, timings, displacements)
                    flux = np.r_[self.case.water_density * s**2 * normal, kw * solution[nw:]]
                    scatter.append(resident.single_receiver @ flux - resident.double_receiver @ solution[:nw])
                    residuals.extend((ice_residual, water_residual))
            scatter = scatter[0] if sources is None else np.column_stack(scatter)
        else:
            factor, load, _, reduced_load, reduced_average = self._factor_condensed_ice(s, timings)
            water, wc, kw, data = self.boundary_system(
                self.water, s, self.case.water_density, receiver, timings=timings, receiver_only=True,
            )
            if self.gpu_trace_library is not None or self.gpu_panel_schur_library is not None:
                solution, displacement, normal, water_residual = self._solve_direct_water(
                    factor, rhs, reduced_load, reduced_average, timings, water=water, coupling=wc,
                    keep_displacement=displacements is not None,
                )
            else:
                solved = self._iterate_water(s, timings, water, wc, rhs, reduced_load, reduced_average, factor)
                solution, displacement, normal = solved.solution, solved.displacement, solved.normal
                water_residual = solved.relative_residual
            ice_residual = self._condensed_ice_residual(s, factor, load, displacement, solution, timings, displacements)
            flux = np.r_[self.case.water_density * s**2 * normal, kw * solution[nw:]]
            scatter = data["S_receiver"] @ flux - data["K_receiver"] @ solution[:nw]
            residuals = (ice_residual, water_residual)
        residual = float(np.max(residuals))
        if not np.isfinite(scatter).all() or not np.isfinite(residual) or residual > RESIDUAL_TOLERANCE:
            raise RuntimeError(f"Zero traction failed original-equation verification at {s}: {residual:g}")
        return direct, {"zero_traction": (scatter, residual)}, False

    def _condensed_ice_residual(self, s, factor, load, displacement, solution, timings, displacements):
        condensed = self.condensation
        nw = self.water.surface_count
        with phase(timings, "ice_original_residual_s"):
            if self.gpu_ice_residual:
                with phase(timings, "ice_panel_gpu_original_residual_s"):
                    ice_residual = factor.original_residual()
            elif condensed is None:
                full_rhs = load @ solution[:nw]
                ice_residual = self.full_ice.relative_residual(factor.matrix, displacement, full_rhs)
            else:
                full_rhs = load @ solution[:nw]
                displacement, ice_residual = condensed.lift_and_check(s, displacement, full_rhs)
        if displacements is not None:
            displacements["zero_traction"] = displacement
        return ice_residual

    def _solve_resident_shots(self, factor, incident, load, average, timings, resident, *, keep_displacement=False):
        """Share both numerical factorizations and recover each shot's elastic field.

        Consume an elastic residual before advancing the iterator because the
        factor's work buffers retain only the most recent recovery solve.
        """
        yield self._solve_direct_water(factor, incident[:, 0], load, average, timings, resident=resident,
                                       keep_displacement=keep_displacement)
        if incident.shape[1] == 1:
            return
        with phase(timings, "zero_coupled_gpu_s"):
            solutions = self.water_direct.solve_prepared_resident(
                resident.boundary, resident.coupling, incident[:, 1:])
        for solution, rhs in zip(solutions.T, incident[:, 1:].T):
            displacement, normal, residual = self._validate_direct_water(
                factor, solution, rhs, load, average, timings, resident=resident,
                keep_displacement=keep_displacement)
            yield solution, displacement, normal, residual

    def _solve_direct_water(self, factor, rhs, load, average, timings, *, water=None, coupling=None, resident=None,
                            keep_displacement=False):
        with phase(timings, "zero_coupled_gpu_s"):
            if self.gpu_water_workspace and self.water_direct is None:
                from .gpu.water import PlanarWaterDirect

                if self.gpu_water_fast_refinement:
                    options = {"fast_refinement": True}
                else:
                    options = {"mixed_precision": True} if self.gpu_water_refinement else {}
                self.water_direct = PlanarWaterDirect(self.water_gpu_library, self.gpu_device,
                                                     self.water.panel_count, self.water.surface_count, **options)
            response = factor.panel_response()
            if resident is not None:
                solution = self.water_direct.solve_resident(resident.boundary, resident.coupling, response, rhs)
            elif self.water_direct is not None:
                solution = self.water_direct.solve(water, coupling, response, rhs)
            else:
                from .gpu.water import solve_coupled_water_gpu

                solution = solve_coupled_water_gpu(self.water_gpu_library, self.gpu_device, water, coupling, response, rhs)
            if self.gpu_water_refinement and timings is not None:
                accepted, iterations, residual = self.water_direct.refinement_stats
                timings["water_refinement_accepted"] = int(accepted)
                if iterations is not None:
                    timings["water_refinement_iterations"] = iterations
                if residual is not None:
                    timings["water_refinement_residual"] = residual
        displacement, normal, residual = self._validate_direct_water(
            factor, solution, rhs, load, average, timings, water=water, coupling=coupling, resident=resident,
            keep_displacement=keep_displacement)
        return solution, displacement, normal, residual

    def _validate_direct_water(self, factor, solution, rhs, load, average, timings, *,
                               water=None, coupling=None, resident=None, keep_displacement=False):
        with phase(timings, "zero_validation_s"):
            pressure = solution[:self.water.surface_count]
            if self.gpu_ice_normal and not keep_displacement:
                displacement = None
                normal = factor.solve_panel_normal(pressure)
            else:
                displacement = factor.solve(load @ pressure)
                normal = average @ displacement
            if resident is not None:
                residual = self.water_direct.residual_resident(
                    resident.boundary, resident.coupling, solution, normal, rhs,
                )
            else:
                diff = water @ solution - coupling @ normal - rhs
                residual = np.linalg.norm(diff) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
        return displacement, normal, residual

    def _iterate_water(self, s, timings, water, wc, rhs, reduced_load, reduced_average, factor):
        from .iterative import WaterConvergenceError, solve_water

        wet = np.flatnonzero(np.diff(reduced_load.indptr)) if self.mumps_library is not None else None

        def solve_ice(force):
            with phase(timings, "ice_iterative_backsolve_s"):
                if timings is not None:
                    timings["ice_iterative_backsolves"] = timings.get("ice_iterative_backsolves", 0) + 1
                if wet is not None:
                    return factor.solve_sparse(wet, force[wet])
                return factor.solve(force)

        thickness = self.geometry.top - self.geometry.bottom
        average_thickness = 0.5 * (thickness[:-1] + thickness[1:])

        def progress(iteration, residual):
            if timings is not None:
                print(f"Water GMRES at {s}: {iteration} iterations, preconditioned residual {residual:.3g}",
                      flush=True)

        iteration_options = dict(s=s, ice_density=self.ice.rho, ice_vp=self.case.ice_p_speed, ice_vs=self.case.ice_s_speed, rtol=2e-10,
                                 ice_thickness=average_thickness, progress=progress,
                                 pressure_x=self.water.centres[:self.water.surface_count, 0],
                                 restart=self.water_gmres_restart)
        if self.water_gpu_library is not None:
            iteration_options.update(gpu_device=self.gpu_device, gpu_library=self.water_gpu_library)
        with phase(timings, "water_iterative_s"):
            try:
                solved = solve_water(water, wc, rhs, reduced_load, reduced_average, solve_ice,
                                     preconditioner=self.water_preconditioner, **iteration_options)
            except WaterConvergenceError:
                if self.water_preconditioner == "spectral":
                    raise
                solved = solve_water(water, wc, rhs, reduced_load, reduced_average, solve_ice,
                                     preconditioner="spectral", **iteration_options)
        if timings is not None:
            timings["water_gmres_iterations"] = solved.iterations
            timings["water_preconditioner_s"] = solved.preconditioner_s
        return solved

    def evaluate(self, s, timings=None, *, sources=None, receivers=None, displacements=None):
        """Solve one Laplace point, sharing the factorization across explicit sources.

        Survey inputs have shape (n, 2). Leave sources unset for a single shot at
        the case source with 1D receiver outputs, including on the condensed path.
        On the zero-traction CPU direct and resident CUDA paths, supply source
        arrays to obtain 2D receiver-by-source outputs in input source order.
        For a single shot, an optional displacements dict receives the original
        elastic displacement vector for each computed air boundary condition.
        """
        condensed = self.condensation is not None or self.full_ice is not None
        if sources is not None:
            if displacements is not None:
                raise ValueError("Elastic displacement capture requires a single shot")
            if (condensed and not self.gpu_resident_bem) or self.air is not None or receivers is None:
                raise ValueError("Explicit sources need receivers and a zero-traction CPU direct or resident CUDA solver")
            sources = self._survey_points(sources)
        if receivers is not None:
            if self.air is not None:
                raise ValueError("Survey receivers require the zero-traction solver")
            receivers = self._survey_points(receivers)
        if condensed:
            return self._evaluate_condensed(s, timings, receivers=receivers, sources=sources,
                                            displacements=displacements)
        case = self.case
        nw = self.water.surface_count
        water, wc, kw, water_data = self.boundary_system(
            self.water, s, self.case.water_density, case.receiver if receivers is None else receivers, timings=timings
        )
        with phase(timings, "source_and_ice_trace_maps_s"):
            if sources is not None:
                columns = [self.water.incident(s, source, receivers) for source in sources]
                rhs, direct = (np.column_stack(values) for values in zip(*columns))
            else:
                rhs, direct = self.water.incident(s, case.source, case.receiver if receivers is None else receivers)
            load_w, avg_w = self.ice.coupling(s, "bottom")
            if self.air:
                load_a, avg_a = self.ice.coupling(s, "top", self.geometry.air_edges)
                load, average = hstack((load_w, load_a), format="csr"), vstack((avg_w, avg_a), format="csr")
            else:
                load, average = load_w, avg_w
        self.ice.timings = timings
        try:
            factor, response, fallback_matrix = self._response(s, load, average, timings)
        finally:
            self.ice.timings = None
        results = {}

        def receive(solution):
            x, normal = solution
            flux = np.r_[self.case.water_density * s**2 * normal, kw * x[nw:]]
            return water_data["S_receiver"] @ flux - water_data["K_receiver"] @ x[:nw]

        def reconstructed_ice(traction):
            displacement = factor.solve(traction)
            if fallback_matrix is not None:
                residual = np.linalg.norm(fallback_matrix @ displacement - traction)
                return displacement, residual / max(np.linalg.norm(traction), np.finfo(float).tiny)
            return displacement, factor.relative_residual(displacement, traction)

        if case.air != "coupled":
            with phase(timings, "zero_coupled_blocks_s"):
                matrix = water.copy(order="F")
                matrix[:, :nw] -= wc @ response[:nw, :nw]
            x = self.dense_solve(matrix, rhs, timings)
            with phase(timings, "zero_validation_s"):
                u, ri = reconstructed_ice(load_w @ x[:nw])
                normal = avg_w @ u
                residual = np.linalg.norm(water @ x - wc @ normal - rhs) / np.linalg.norm(rhs)
                results["zero_traction"] = (receive((x, normal)), max(residual, ri))
                if displacements is not None:
                    displacements["zero_traction"] = u

        if self.air:
            air, ac, _, _ = self.boundary_system(self.air, s, self.case.air_density, timings=timings, prefix="air")
            na = self.air.surface_count
            n_water = water.shape[0]
            size = n_water + air.shape[0]
            with phase(timings, "air_coupled_blocks_s"):
                matrix = np.zeros((size, size), dtype=np.complex128, order="F")
                matrix[:n_water, :n_water] = water
                matrix[n_water:, n_water:] = air
                matrix[:n_water, :nw] -= wc @ response[:nw, :nw]
                matrix[:n_water, n_water:n_water + na] = -wc @ response[:nw, nw:]
                matrix[n_water:, :nw] = -ac @ response[nw:, :nw]
                matrix[n_water:, n_water:n_water + na] -= ac @ response[nw:, nw:]
            solution = self.dense_solve(matrix, np.r_[rhs, np.zeros(air.shape[0], dtype=complex)], timings, "air")
            with phase(timings, "air_validation_s"):
                x, y = solution[:n_water], solution[n_water:]
                u, ri = reconstructed_ice(load_w @ x[:nw] + load_a @ y[:na])
                nw_u, na_u = avg_w @ u, avg_a @ u
                residual_water = np.linalg.norm(water @ x - wc @ nw_u - rhs) / np.linalg.norm(rhs)
                residual_air = np.linalg.norm(air @ y - ac @ na_u)
                air_scale = np.linalg.norm(ac @ na_u)
                residual_air /= max(air_scale, np.finfo(float).tiny)
                results["coupled"] = (receive((x, nw_u)), max(residual_water, residual_air, ri))
                if displacements is not None:
                    displacements["coupled"] = u

        for name, (value, residual) in results.items():
            if not np.isfinite(value).all() or not np.isfinite(residual) or residual > RESIDUAL_TOLERANCE:
                raise RuntimeError(f"{name} failed residual check at {s}: {residual:g}")
        return direct, results, fallback_matrix is not None

    def _survey_points(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("Survey coordinates must form a nonempty (n, 2) array")
        x, z = points.T
        g = self.geometry
        if not np.all((g.low < x) & (x < g.high) & (-self.case.depth < z)
                      & (z < np.interp(x, g.edges, g.bottom))):
            raise ValueError("Survey points must lie inside the physical water domain")
        return points
