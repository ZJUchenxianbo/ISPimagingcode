"""Run small physical/sensitivity checks without retaining temporary wavefields.

Usage: .venv/bin/python -m ice_imaging_2d.experiments.validation_checks
"""

import argparse
import json
from pathlib import Path
import tempfile

from ..ice2d.dual_config import read_dual_config
from ..ice2d.checks import plane_layer_checks, sensitivity_checks, acquisition_checks, local_reconstruction_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON summary; wavefields are always temporary")
    args = parser.parse_args()
    config = Path(__file__).parents[1] / "configs/dual_smoke.json"
    physics, inverse, _ = read_dual_config(config)
    with tempfile.TemporaryDirectory(prefix="ice_validation_") as directory:
        result = {}
        for name, function in (
            ("plane_layer", lambda: plane_layer_checks()),
            ("acquisition", lambda: acquisition_checks(physics, directory)),
            ("sensitivity", lambda: sensitivity_checks(physics, inverse, directory)),
            ("local_reconstruction", lambda: local_reconstruction_check(physics, inverse, directory)),
        ):
            result[name] = function()
            print(name, json.dumps(result[name], indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    if not all(value["passed"] for value in result.values()):
        raise SystemExit("Numerical checks failed; inspect diagnostics above")


if __name__ == "__main__":
    main()
