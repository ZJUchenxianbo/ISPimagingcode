"""Inversion only: measured first-reflection Cauchy traces to paper TD-DSM.

No forward solver, GPU library or mesh refinement is imported or invoked here.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .data_store import exclusive_run, identifier, write_manifest
from .geometry import expression
from .imaging import dsm_energy, image_grid
from .metrics import evaluate, extract_ridge
from .plotting import plot_case
from .processing import process_records


METHODS = ('dsm', 'dsm_corrected', 'correction_only')
METHOD_VERSION = 'first-reflection-td-dsm-v2'


def invert(data, experiment, name, output, *, data_path, data_sha256):
    config = experiment.imaging()
    factors = experiment.values['imaging']['window_factors']
    identity = dict(method=METHOD_VERSION, input_sha256=data_sha256,
                    imaging=asdict(config), window_factors=factors)
    run_id = identifier(identity)
    directory = Path(output) / 'inversion' / name / run_id
    with exclusive_run(directory):
        metadata_path = directory / 'metadata.json'
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text(encoding='utf-8'))
            if saved.get('state') == 'complete' and saved.get('inversion_id') == run_id:
                return dict(case=name, result=str(directory), reused=True,
                            metrics=json.loads((directory / 'metrics.json').read_text(encoding='utf-8')))
        started = perf_counter()
        processed = process_records(data, config)
        images = dsm_energy(processed, data, config)
        x, z = image_grid(config)
        images.update(x=x, z=z)
        # The observed geometry is accessed only AFTER indicators are computed.
        actual = json.loads(str(data['config_json']))['geometry']['bottom']
        truth = expression(actual, x)
        metrics = {method: evaluate(images[method], x, z, truth, config.mean_bottom) for method in METHODS}
        reference_rmse = float(np.sqrt(np.mean((truth - config.mean_bottom)**2)))
        for method in METHODS:
            metrics[method]['maximum_indicator'] = float(images[method].max())
            metrics[method]['improves_reference_rmse'] = metrics[method]['rmse_m'] < reference_rmse
        np.savez_compressed(directory / 'images.npz', **images, true_bottom=truth,
                            **{method + '_ridge': extract_ridge(images[method], z)[0] for method in METHODS})
        mask = processed['receiver_mask'][..., None]
        full_energy = np.sum(processed['pressure']**2, axis=-1)
        kept_energy = np.sum(processed['gated_pressure']**2, axis=-1)
        energy_fraction = np.divide(kept_energy, full_energy, out=np.zeros_like(kept_energy),
                                    where=full_energy > 0)
        np.savez_compressed(directory / 'first_reflection.npz', time=data['time'],
                            receiver_mask=processed['receiver_mask'], window=processed['window'],
                            v_chi_first=processed['gated_pressure'] * mask,
                            dn_v_chi_first=-processed['gated_gradient'] * mask,
                            normal='downward', reference_arrival=processed['reference_arrival'],
                            test_pulse=processed['source_signal'],
                            pressure_energy_fraction=energy_fraction)
        plot_case(directory, data, processed, images, metrics, truth, config)
        window_checks = []
        for factor in factors:
            if factor == 1:
                candidate = images
            else:
                settings = replace(config, first_window_halfwidth=config.first_window_halfwidth * factor)
                candidate = dsm_energy(process_records(data, settings), data, settings)
            window_checks.append(dict(factor=factor, metrics={
                method: evaluate(candidate[method], x, z, truth, config.mean_bottom) for method in METHODS}))
        source = processed['source_signal']
        inside = abs(data['time'] - config.source_center) <= config.first_window_halfwidth
        metadata = dict(**identity, state='complete', inversion_id=run_id, case=name,
                        input_data=str(Path(data_path).resolve()), forward_id=str(data['forward_id']),
                        normal='measurement line downward; recorded +z derivative negated',
                        symbols=dict(dsm='I_0^(1)', dsm_corrected='I^(1)', correction_only='I_corr'),
                        data_window='receiver time', integration_window='positive correlation lag [0,T)',
                        reference_rmse_m=reference_rmse, mirror_plane_z=config.mean_bottom,
                        transmission_focusing_proved=False, upper_interface_inverted=False,
                        receiver_pairs_kept=int(np.sum(processed['receiver_mask'])),
                        receiver_pairs_total=int(processed['receiver_mask'].size),
                        mean_kept_pressure_energy_fraction=float(np.mean(
                            energy_fraction[processed['receiver_mask'].astype(bool)])),
                        filtered_source_energy_outside_halfwindow=float(np.sum(source[~inside]**2) / np.sum(source**2)),
                        wall_seconds=perf_counter() - started,
                        note='Time gating is approximate first-reflection separation, not exact mode separation.')
        write_manifest(directory / 'metrics.json', metrics)
        write_manifest(directory / 'window_sensitivity.json', window_checks)
        write_manifest(metadata_path, metadata)
    return dict(case=name, result=str(directory), reused=False, metrics=metrics)
