"""CQ-BEM/Q4-FEM observations and first-reflection TD-DSM of the ice bottom."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('-forward', '--forward', dest='forward_only', action='store_true',
                        help='Only generate/save Cauchy observations; resume matching CQ checkpoints')
    stages.add_argument('-inv', '--inv', dest='inverse_only', action='store_true',
                        help='Only invert saved observations; never call a PDE solver or require CUDA')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'configs/experiment.json')
    parser.add_argument('--case', help='One configured case; default: all configured cases')
    parser.add_argument('--output', type=Path, default=Path('ice_outputs/ice_bottom'))
    parser.add_argument('--data', type=Path, help='-inv only: observation data root, case directory, or NPZ file')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), help='Forward execution backend override')
    parser.add_argument('--workers', type=int, help='CPU processes; CUDA workers per device are configured in JSON')
    parser.add_argument('--estimate', action='store_true', help='Validate and report configuration/sizes without solving')
    args = parser.parse_args(argv)
    if args.inverse_only and (args.backend is not None or args.workers is not None):
        parser.error('-inv accepts no --backend/--workers: it does not run a forward solver')
    if args.data is not None and not args.inverse_only:
        parser.error('--data belongs to -inv')
    if args.workers is not None and args.workers < 1:
        parser.error('--workers must be positive')
    from ice_imaging_2d.ice2d.workflow import run_experiment
    try:
        return run_experiment(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
