"""Bounded kernel timing; records are never used as imaging observations."""
import argparse
import json
import time
import resource
from pathlib import Path
import numpy as np
from ..ice2d.dual_config import read_dual_config
from ..ice2d.model import make_model
from ..ice2d.solver import _advance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    config, _, _ = read_dual_config(args.config)
    started = time.perf_counter()
    model = make_model(config, 'sine')
    build = time.perf_counter() - started
    def run(steps):
        return _advance(model['lam'], model['mu'], model['corner_mu'],
                        model['inverse_density_x'], model['inverse_density_z'], model['damping'],
                        model['dt'], config.grid_spacing, np.ones(steps, np.float32),
                        model['source_nodes'][0], model['source_weights'][0],
                        model['receiver_nodes'], model['receiver_weights'], model['stride'],
                        np.empty(0, dtype=np.int64), config.spatial_order)
    run(1)
    started = time.perf_counter()
    run(args.steps)
    elapsed = time.perf_counter() - started
    result = dict(config=args.config, steps=args.steps, build_seconds=build,
                  kernel_seconds=elapsed, estimated_full_shot_seconds=elapsed / args.steps * model['steps'],
                  peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  note='Kernel extrapolation only, not full-waveform validation; RSS KiB on Linux, bytes on macOS')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
