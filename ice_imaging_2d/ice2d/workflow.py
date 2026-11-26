"""Independent forward and inverse stages, sharing immutable Cauchy data only."""

import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data_store import exclusive_run, file_digest, write_manifest
from .forward_cq import SOLVER_VERSION
from .io import load_data, validate_cauchy


def observation_path(base, name, single_case):
    base = Path(base)
    if base.is_file() or (base / 'data_cq.npz').is_file():
        if not single_case:
            raise ValueError('A single observation file/directory requires selecting one --case')
        return base if base.is_file() else base / 'data_cq.npz'
    return base / name / 'data_cq.npz'


def read_observation(path, forward):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Observation missing: {path}; run -forward first (inversion never generates data)')
    data = validate_cauchy(load_data(path))
    if str(data.get('solver_version', '')) != SOLVER_VERSION:
        raise ValueError('Observation solver version is incompatible with this pipeline')
    if str(data.get('forward_id', '')) != forward.signature():
        raise ValueError('Observation physics/acquisition/CQ signature differs; select matching data/config')
    recorded = json.loads(str(data['config_json']))
    from .forward_cq.config import ForwardConfig
    if ForwardConfig.from_dict(recorded).signature() != forward.signature():
        raise ValueError('Observation configuration and declared identity disagree')
    grid = forward.grid()
    for name, expected in (('source_x', forward.sources[:, 0]), ('source_z', forward.sources[:, 1]),
                           ('receiver_x', forward.receivers[:, 0]), ('time', grid.time_s[:grid.n_out])):
        if data[name].shape != expected.shape or not np.allclose(data[name], expected, rtol=0, atol=1e-12):
            raise ValueError(f'Observation {name} disagrees with declared coordinates')
    if not np.isclose(float(data['array_z']), forward.receivers[0, 1], rtol=0, atol=1e-12):
        raise ValueError('Observation array depth disagrees with configuration')
    return data


def generate_observation(forward, directory, workers):
    directory = Path(directory)
    path = directory / 'data_cq.npz'
    with exclusive_run(directory):
        if path.is_file():
            data = read_observation(path, forward)
            if not (directory / 'metadata.json').is_file():
                write_manifest(directory / 'metadata.json', json.loads(str(data['metadata_json'])))
            return path, True
        # No forward implementation or CUDA library is imported in the inverse path.
        from .forward_cq.adapter import simulate, save_data
        data, metadata = simulate(forward, workers=workers, checkpoint_dir=directory / 'cq_cache')
        validate_cauchy(data)
        save_data(directory, data, metadata)
        return path, False


def run_experiment(args):
    experiment = ExperimentConfig.read(args.config)
    if args.backend is not None:
        experiment.values['forward'].setdefault('solver', {})['backend'] = args.backend
        experiment.validate()
    names = [args.case] if args.case else list(experiment.values['cases'])
    if any(name not in experiment.values['cases'] for name in names):
        raise ValueError(f'Unknown case; choices: {list(experiment.values["cases"])}')
    mode = 'forward' if args.forward_only else 'inverse' if args.inverse_only else 'all'
    output = Path(args.output)
    if args.estimate:
        report = dict(stage=mode, carrier_hz=15000, imaging_band_hz=[10000, 20000],
                      roughness=experiment.values['roughness'], cases=experiment.values['cases'],
                      imaging=experiment.values['imaging'])
        if mode != 'inverse':
            report['forward'] = {name: experiment.for_case(name).summary() for name in names}
        else:
            report['data_root'] = str(args.data or output / 'data')
            report['requires_cuda'] = False
        print(json.dumps(report, indent=2), flush=True)
        return report
    results = []
    for name in names:
        forward = experiment.for_case(name)
        if mode != 'inverse':
            print(json.dumps(dict(case=name, forward=forward.summary()), indent=2), flush=True)
            path, reused = generate_observation(forward, output / 'data' / name, args.workers or 2)
            if mode == 'forward':
                results.append(dict(case=name, data=str(path.resolve()), forward_reused=reused))
                continue
        else:
            path = observation_path(args.data or output / 'data', name, len(names) == 1)
        data = read_observation(path, forward)
        from .bottom_stage import invert
        result = invert(data, experiment, name, output, data_path=path, data_sha256=file_digest(path))
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    print(json.dumps(dict(stage=mode, output=str(output.resolve()), results=results), indent=2), flush=True)
    return results
