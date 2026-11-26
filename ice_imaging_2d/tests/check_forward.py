"""Lightweight numerical checks of migrated CQ forward data (no neural networks).

Run with --gpu on Linux after building the CUDA libraries for CPU/GPU comparison.
The ordinary CPU checks do not import or initialize CUDA.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
from scipy.special import kv

from ..ice2d.forward_cq.coupled import Case, PlanarTransfer
from ..ice2d.forward_cq.config import ForwardConfig
from ..ice2d.forward_cq.adapter import create_model, evaluate_survey, simulate, _close_model
from .fixtures import coarse_forward, coarse_imaging


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-30))


def run(gpu=False):
    s = 160 + 942j
    case = Case(2, 2.5, .5, .5, '0', '-0.7', (-.3, -1.5), (.37, -1.43),
                150, 'zero-traction', water_panel_m=.7)
    model = PlanarTransfer(case, 200)
    scalar = PlanarTransfer(replace(case, cauchy=False), 200)
    direct, results, _ = model.evaluate(s)
    pressure, gradient = results['zero_traction'][0]
    errors = []
    for step in (.002, .001, .0005):
        plus = scalar.evaluate(s, receivers=[[.37, -1.43 + step]])
        minus = scalar.evaluate(s, receivers=[[.37, -1.43 - step]])
        difference = (plus[1]['zero_traction'][0] - minus[1]['zero_traction'][0]) / (2 * step)
        errors.append(relative(gradient, difference))
    assert errors[-1] < 1e-6 and errors[-1] < errors[0] / 10, errors
    dx, dz = np.asarray(case.receiver) - case.source
    radius = np.hypot(dx, dz)
    exact = np.array([kv(0, s * radius / case.water_speed),
                      -s / case.water_speed * kv(1, s * radius / case.water_speed) * dz / radius]) / (2 * np.pi)
    assert relative(direct, exact) < 1e-13

    # Fixture evaluated from the inspected original CPU definitions in the ZIP,
    # with rho_ice=900 and air_speed=340.3; this tests the migrated pressure path.
    reference_case = replace(case, receiver=(.4, -1.5), ice_density=900, air_speed=340.3, cauchy=False)
    reference = PlanarTransfer(reference_case, 200).evaluate(s)[1]['zero_traction'][0]
    expected = 0.044152620325438346 + 0.04727987460137367j
    assert relative(reference, expected) < 1e-9

    sources = np.array([[-.3, -1.5], [.3, -1.5]])
    receivers = np.array([[-.6, -1.5], [0, -1.5], [.6, -1.5]])
    batch = model.evaluate(s, sources=sources, receivers=receivers)[1]['zero_traction'][0]
    serial = np.empty_like(batch)
    for i, source in enumerate(sources):
        model.case = replace(case, source=tuple(source))
        serial[:, i] = model.evaluate(s, receivers=receivers)[1]['zero_traction'][0]
    model.case = case
    batch_error = relative(batch, serial)
    assert batch_error < 1e-10

    air = PlanarTransfer(replace(case, air='coupled'), 200)
    air_values, air_responses, _ = air.evaluate(s)
    assert np.isfinite(air_values).all() and air_responses['coupled'][1] < 2e-8
    air_scalar = PlanarTransfer(replace(case, air='coupled', cauchy=False), 200)
    step = .001
    air_scalar.case = replace(air_scalar.case, receiver=(.37, -1.43 + step))
    plus = air_scalar.evaluate(s)[1]['coupled'][0]
    air_scalar.case = replace(air_scalar.case, receiver=(.37, -1.43 - step))
    minus = air_scalar.evaluate(s)[1]['coupled'][0]
    air_gradient_error = relative(air_responses['coupled'][0][1], (plus - minus) / (2 * step))
    assert air_gradient_error < 1e-5

    config = coarse_forward()
    data, metadata = simulate(config, workers=1)
    assert metadata['source_projection_relative_error'] < .001
    assert data['pressure'].shape == (2, 3, len(data['time']))
    assert relative(data['pressure'], data['pressure_direct'] + data['pressure_scattered']) == 0
    assert relative(data['gradient'], data['gradient_direct'] + data['gradient_scattered']) == 0
    assert np.isfinite(data['pressure']).all() and np.isfinite(data['gradient']).all()
    geometry = json.loads(data['geometry_json'])
    assert len(geometry['nodes']) == len(geometry['bottom'])

    # Data explicitly carries the separated direct/scattered fields and source
    # convention, so first-reflection processing must not subtract direct twice.
    from ..ice2d.processing import process_records
    from ..ice2d.imaging import dsm_energy
    image_config = coarse_imaging()
    processed = process_records(data, image_config)
    images = dsm_energy(processed, data, image_config)
    assert all(np.isfinite(image).all() for image in images.values())

    # A small independent CQ time refinement of the analytic free-water transfer.
    coarse_grid = config.grid()
    values = deepcopy(config.values)
    values['frequency']['dt_s'] /= 2
    fine_grid = ForwardConfig(values).grid()
    distance = np.linalg.norm(config.sources[0] - config.receivers[0])
    def trace(grid):
        return grid.waveform(kv(0, grid.points[grid.selected] * distance / 1500) / (2 * np.pi))
    coarse, fine = trace(coarse_grid), trace(fine_grid)[::2]
    cq_change = relative(coarse, fine)
    assert cq_change < .3  # coarse smoke grid only; report the actual difference

    gpu_report = 'not_run'
    if gpu:
        config.values['solver']['backend'] = 'gpu'
        config.validate()
        cpu_values = deepcopy(config.values)
        cpu_values['solver']['backend'] = 'cpu'
        cpu_model = create_model(ForwardConfig(cpu_values))
        gpu_model = create_model(config, device=config.values['solver']['gpu_devices'][0])
        try:
            grid = config.grid()
            checks = []
            for point in grid.points[grid.selected][[0, len(grid.selected) // 2, -1]]:
                cpu_result = evaluate_survey(cpu_model, config, point)
                gpu_result = evaluate_survey(gpu_model, config, point)
                error = relative(gpu_result[1], cpu_result[1])
                assert error < 1e-6 and gpu_result[2] < 2e-8
                checks.append(error)
            gpu_report = dict(single_point_cauchy_relative_errors=checks)
        finally:
            _close_model(cpu_model)
            _close_model(gpu_model)
        gpu_data, gpu_metadata = simulate(config, workers=1)
        pressure_error = relative(gpu_data['pressure'], data['pressure'])
        gradient_error = relative(gpu_data['gradient'], data['gradient'])
        assert pressure_error < 1e-6 and gradient_error < 1e-6
        gpu_report.update(cq_pressure_relative_error=pressure_error,
                          cq_gradient_relative_error=gradient_error,
                          cpu_wall_seconds=metadata['wall_seconds'],
                          gpu_wall_seconds=gpu_metadata['wall_seconds'])
    return dict(upstream_fixture_relative_error=relative(reference, expected),
                normal_derivative_step_errors=errors, direct_gradient_relative_error=relative(direct, exact),
                batch_relative_error=batch_error, coupled_air_residual=air_responses['coupled'][1],
                coupled_air_gradient_relative_error=air_gradient_error,
                smoke_max_residual=metadata['max_residual'], free_water_cq_dt_refinement_change=cq_change,
                gpu=gpu_report, neural_network='skipped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Run CUDA comparisons on a prepared Linux GPU host')
    args = parser.parse_args()
    print(json.dumps(run(args.gpu), indent=2))
