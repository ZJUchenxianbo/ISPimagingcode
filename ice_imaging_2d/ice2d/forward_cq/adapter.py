"""CPU/CUDA CQ survey orchestration and the project's Cauchy NPZ contract."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
import multiprocessing
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy

from ..data_store import identifier, write_manifest
from .config import ForwardConfig
from . import SOLVER_VERSION
from .coupled import PlanarTransfer, make_geometry


_MODEL = None
_CONFIG = None


def gpu_options(config, device):
    solver = config.values['solver']
    if solver['backend'] == 'cpu':
        return dict(zero_solver='direct')
    if platform.system() != 'Linux':
        raise ValueError('The supplied CUDA backend requires Linux; CPU is available on Windows')
    directory = Path(solver['gpu_library_dir'] or Path(__file__).with_name('gpu')).resolve()
    names = dict(bem_gpu_library='libplanar_bem_cuda_v7b.so',
                 gpu_panel_schur_library='libgpu_panel_schur.panel-normal.so',
                 water_gpu_library='libplanar_water_tangent.so')
    libraries = {key: str(directory / name) for key, name in names.items()}
    missing = [path for path in libraries.values() if not Path(path).is_file()]
    if missing:
        raise ValueError(f'Missing CUDA libraries: {missing}; run forward_cq/gpu/build.sh first')
    return dict(**libraries, zero_solver='condensed', gpu_device=device,
                gpu_trace_full_ice=True, gpu_resident_bem=True, gpu_water_workspace=True,
                gpu_ice_assembly=True, gpu_ice_normal=True, gpu_ice_residual=True)


def create_model(config, *, device=0, cauchy=True):
    grid = config.grid()
    model = PlanarTransfer(config.case(cauchy=cauchy), grid.design_hz, **gpu_options(config, device))
    # Validate every acquisition point before allocating time histories.
    model._survey_points(config.sources)
    model._survey_points(config.receivers)
    return model


def _close_model(model):
    if model is not None:
        for name in ('ice_factor', 'water_direct'):
            handle = getattr(model, name, None)
            if handle is not None and hasattr(handle, 'close'):
                handle.close()


def gpu_environment(config):
    if config.values['solver']['backend'] != 'gpu':
        return None
    results = {}
    cuda_root = os.environ.get('CUDA_ROOT', '/usr/local/cuda')
    for name, command in (
            ('nvcc', [str(Path(cuda_root) / 'bin/nvcc'), '--version']),
            ('nvidia_smi', ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
            results[name] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            results[name] = f'unavailable: {type(error).__name__}'
    return results


def _initialize(values, devices, next_worker):
    global _MODEL, _CONFIG
    _CONFIG = ForwardConfig(values).validate()
    with next_worker.get_lock():
        device = devices[next_worker.value % len(devices)]
        next_worker.value += 1
    _MODEL = create_model(_CONFIG, device=device)


def evaluate_survey(model, config, s):
    """Return [pressure rows, +z derivative rows] x sources at one CQ point."""
    timings = {}
    if model.case.air == 'zero-traction':
        direct, response, fallback = model.evaluate(
            s, timings, sources=config.sources, receivers=config.receivers)
        scattered, residual = response['zero_traction']
        return direct, scattered, residual, int(fallback), timings
    # The upstream explicit-air branch has no survey factorization API.
    # Keep its verified scalar solve and make the extra solve cost explicit.
    nr, ns = len(config.receivers), len(config.sources)
    count = 2 * nr if model.case.cauchy else nr
    direct, scattered = np.empty((count, ns), complex), np.empty((count, ns), complex)
    residual, fallbacks = 0.0, 0
    original = model.case
    try:
        for source_index, source in enumerate(config.sources):
            for receiver_index, receiver in enumerate(config.receivers):
                model.case = replace(original, source=tuple(source), receiver=tuple(receiver))
                values, responses, fallback = model.evaluate(s, timings)
                signal, error = responses['coupled']
                rows = [receiver_index, nr + receiver_index] if original.cauchy else [receiver_index]
                direct[rows, source_index] = np.atleast_1d(values)
                scattered[rows, source_index] = np.atleast_1d(signal)
                residual = max(residual, error)
                fallbacks += int(fallback)
    finally:
        model.case = original
    return direct, scattered, residual, fallbacks, timings


def _solve_point(job):
    row, s = job
    return row, evaluate_survey(_MODEL, _CONFIG, s)


def simulate(config, *, workers=1, checkpoint_dir=None):
    """Generate a complete survey; CPU workers or GPU workers per configured card."""
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    config.validate()
    grid = config.grid()
    source_error = config.source_projection_error(grid)
    if source_error is not None and source_error > 0.05:
        raise ValueError(f'CQ band distorts the Gaussian source by {source_error:.1%}; '
                         'widen the reconstruction band before running. Processing band is independent.')
    solver = config.values['solver']
    devices = solver['gpu_devices'] if solver['backend'] == 'gpu' else [0]
    gpu_options(config, devices[0])  # reject unavailable CUDA before allocating survey arrays
    process_count = (len(devices) * solver['workers_per_device'] if solver['backend'] == 'gpu' else workers)
    process_count = min(process_count, len(grid.selected))
    shape = (len(grid.selected), 2 * len(config.receivers), len(config.sources))
    direct, scattered = np.empty(shape, complex), np.empty(shape, complex)
    residuals = np.empty(len(grid.selected))
    phase_timings, fallbacks = {}, 0
    start = perf_counter()
    mesh = None
    checkpoint_dir = Path(checkpoint_dir) / config.signature() if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reused = 0
    cache_backends = set()

    def accept(row, result, *, cached=False):
        nonlocal fallbacks
        direct[row], scattered[row], residuals[row], fallback, timings = result
        if not (np.isfinite(direct[row]).all() and np.isfinite(scattered[row]).all()
                and np.isfinite(residuals[row]) and residuals[row] <= 2e-8):
            raise ValueError(f'Invalid or unconverged CQ point {row}')
        fallbacks += fallback
        for name, value in timings.items():
            if name.endswith('_bytes'):
                phase_timings[name] = max(phase_timings.get(name, 0.0), value)
            else:
                phase_timings[name] = phase_timings.get(name, 0.0) + value
        if checkpoint_dir is not None and not cached:
            target = checkpoint_dir / f'{row:06d}.npz'
            temporary = target.with_suffix('.partial.npz')
            np.savez_compressed(temporary, direct=direct[row], scattered=scattered[row],
                                residual=residuals[row], fallback=fallback,
                                point=grid.points[grid.selected[row]], backend=solver['backend'])
            temporary.replace(target)

    jobs = []
    for row, point in enumerate(grid.points[grid.selected]):
        path = checkpoint_dir / f'{row:06d}.npz' if checkpoint_dir is not None else None
        if path is not None and path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if (complex(saved['point']) != point or saved['direct'].shape != shape[1:]
                        or saved['scattered'].shape != shape[1:]):
                    raise ValueError(f'CQ checkpoint mismatch: {path}')
                accept(row, (saved['direct'], saved['scattered'], float(saved['residual']),
                             int(saved['fallback']), {}), cached=True)
                cache_backends.add(str(saved['backend']))
            reused += 1
        else:
            jobs.append((row, point))
    if jobs and process_count == 1:
        model = create_model(config, device=devices[0])
        try:
            mesh = model.mesh_summary()
            for complete, (row, s) in enumerate(jobs, 1):
                accept(row, evaluate_survey(model, config, s))
                if complete == 1 or complete % 10 == 0:
                    print(f'CQ points {complete}/{len(jobs)} (cached {reused})', flush=True)
        finally:
            _close_model(model)
    elif jobs:
        # spawn avoids inheriting CUDA state and also works on Windows.
        context = multiprocessing.get_context('spawn')
        counter = context.Value('i', 0)
        with ProcessPoolExecutor(process_count, mp_context=context, initializer=_initialize,
                                 initargs=(config.values, devices, counter)) as pool:
            tasks = [pool.submit(_solve_point, job) for job in jobs]
            for completed, task in enumerate(as_completed(tasks), 1):
                row, result = task.result()
                accept(row, result)
                if completed == 1 or completed % 10 == 0:
                    print(f'CQ points {completed}/{len(jobs)}', flush=True)

    # waveform's first axis is time; the project stores source x receiver x time.
    direct_time = grid.waveform(direct).transpose(2, 1, 0)
    scattered_time = grid.waveform(scattered).transpose(2, 1, 0)
    nr = len(config.receivers)
    time = grid.time_s[:grid.n_out]
    geometry = make_geometry(config.case(), grid.design_hz)
    data = dict(time=time, source_x=config.sources[:, 0], source_z=config.sources[:, 1],
                receiver_x=config.receivers[:, 0], array_z=float(config.receivers[0, 1]),
                pressure_direct=direct_time[:, :nr], gradient_direct=direct_time[:, nr:],
                pressure_scattered=scattered_time[:, :nr], gradient_scattered=scattered_time[:, nr:],
                solver_version=SOLVER_VERSION, config_json=json.dumps(config.values, sort_keys=True),
                geometry_json=json.dumps(dict(nodes=geometry.edges.tolist(),
                                              bottom=geometry.bottom.tolist(), top=geometry.top.tolist())),
                source_equation='wave_equation_rhs', gradient_normal='positive_z',
                forward_id=config.signature(),
                laplace_points=grid.points[grid.selected], cq_offsets=grid.bins[grid.selected],
                direct_transfer=direct, scattered_transfer=scattered, residuals=residuals,
                source_signal=grid.waveform(np.ones(len(grid.selected), complex)))
    data['pressure'] = data['pressure_direct'] + data['pressure_scattered']
    data['gradient'] = data['gradient_direct'] + data['gradient_scattered']
    metadata = dict(solver_version=SOLVER_VERSION, config=config.values,
                    config_id=identifier(config.values), backend=solver['backend'],
                    forward_id=config.signature(), cached_points=reused, cached_backends=sorted(cache_backends),
                    gradient_normal='positive_z', gradient_method='analytic_receiver_layer_derivative_cpu',
                    source_equation='(c_w^-2 d_tt - Laplacian)p = q(t) delta(x-y)',
                    source_normalization='unit_2d_line_wave_equation_rhs; not legacy stress_rate_rhs',
                    source_projection_relative_error=source_error,
                    lower_boundary='open_water_pml', air=solver['air'], workers=process_count,
                    gpu_devices=devices if solver['backend'] == 'gpu' else [],
                    gpu_cuda_arch=solver['cuda_arch'] or os.environ.get('PLANAR_CUDA_ARCH'),
                    gpu_library_dir=solver['gpu_library_dir'],
                    gpu_environment=gpu_environment(config) if jobs else None,
                    max_residual=float(residuals.max()), elastic_lu_fallback_count=fallbacks,
                    wall_seconds=perf_counter() - start, phase_totals=phase_timings,
                    mesh=mesh or dict(design_hz=grid.design_hz, note='Mesh statistics available in forward --estimate'),
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    gpu_validation='not_certified_by_cpu_checks',
                    note='Finite-band CQ projection; refine band, CQ time step and spatial mesh for fidelity.')
    data['metadata_json'] = json.dumps(metadata, sort_keys=True)
    return data, metadata


def save_data(directory, data, metadata):
    """Never overwrite a previous scientific run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'data_cq.npz'
    if target.exists() or (directory / 'metadata.json').exists():
        raise ValueError('CQ output already exists; choose a new output directory')
    temporary = target.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **data)
    temporary.replace(target)
    write_manifest(directory / 'metadata.json', metadata)
    return target
