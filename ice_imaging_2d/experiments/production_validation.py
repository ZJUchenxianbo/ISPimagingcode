"""Explicit full-frequency comparison of two independently configured forwards.

Run on server: each configuration retains its specified physical geometry.
No automatic promotion of a candidate to production follows this check.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ..ice2d.dual_config import read_dual_config
from ..ice2d.sensitivity import CoupledForward, WaveformObjective
from ..ice2d.data_store import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-config', required=True)
    parser.add_argument('--candidate-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    p, inv, exp = read_dual_config(args.reference_config)
    q, other, other_exp = read_dual_config(args.candidate_config)
    fixed = ('mean_bottom', 'ice_top', 'rough_amplitude', 'rough_period', 'source_frequency',
             'band_low', 'band_high', 'array_depth', 'source_count', 'source_halfwidth',
             'receiver_halfwidth', 'receiver_spacing', 'source_sigma', 'source_center')
    if any(getattr(p, key) != getattr(q, key) for key in fixed) or exp.top_shift != other_exp.top_shift:
        raise ValueError('Comparison must retain physical geometry, source and acquisition')
    from ..ice2d.geometry import Geometry
    from ..ice2d.model import ice_bottom
    from ..ice2d.solver import simulate
    x = np.linspace(inv.control_x[0], inv.control_x[-1], 257)
    geometry = Geometry(p.ice_top + exp.top_shift, tuple(x), tuple(ice_bottom(x, p, 'sine')))
    reference = simulate(p, 'flat', output / 'cache', geometry=geometry)
    candidate = simulate(q, 'flat', output / 'cache', geometry=geometry)
    objective = WaveformObjective(p, inv, reference, candidate)
    residual = objective.residual(candidate)
    baseline = objective.filter(reference['pressure'])
    values = {}
    for window in ('bottom', 'internal'):
        w = objective.weight(window)
        values[window] = float(np.sqrt(np.sum(w * residual**2) / max(np.sum(w * baseline**2), 1e-30)))
    write_manifest(output / 'metadata.json', dict(relative_waveform_errors=values,
                   waveform_gate_passed=all(v <= .03 for v in values.values()),
                   production_approved=False,
                   note='Additional per-trace phase, boundary and sensitivity checks required'))
    print(json.dumps(values, indent=2))


if __name__ == '__main__':
    main()
