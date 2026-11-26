"""Small CLI integration checks; temporary data are removed on completion."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from ..main import main
from ..ice2d.config import ExperimentConfig
from ..ice2d.forward_cq.adapter import simulate, evaluate_survey
from ..ice2d.metrics import extract_ridge
from ..ice2d.workflow import read_observation
from .fixtures import small_experiment


def reject(call):
    try:
        call()
    except (ValueError, SystemExit):
        return
    raise AssertionError('Expected rejection')


def run():
    experiment = small_experiment()
    formal = ExperimentConfig.read(Path(__file__).resolve().parents[1] / 'configs/experiment.json')
    assert formal.forward.values['frequency']['carrier_hz'] == 15000
    assert len(formal.forward.sources) == 121 and len(formal.forward.receivers) == 160
    for section, key, value in [('frequency', 'carrier_hz', 5000), ('processing', 'band_low', 3000)]:
        invalid = deepcopy(experiment.values)
        invalid['forward'][section][key] = value
        reject(lambda: ExperimentConfig.from_dict(invalid))
    for mean in (0, 4.01):
        invalid = deepcopy(experiment.values)
        invalid['cases']['flat']['mean_thickness_m'] = mean
        reject(lambda: ExperimentConfig.from_dict(invalid))
    invalid = deepcopy(experiment.values)
    invalid['cases']['flat'] = dict(kind='sinusoidal', mean_thickness_m=4.)
    reject(lambda: ExperimentConfig.from_dict(invalid))
    invalid['cases']['flat']['kind'] = 'ridge'
    reject(lambda: ExperimentConfig.from_dict(invalid))
    reject(lambda: extract_ridge(np.zeros((3, 3)), np.arange(3)))
    with contextlib.redirect_stderr(io.StringIO()):
        reject(lambda: main(['-forward', '-inv']))
        reject(lambda: main(['-inv', '--backend', 'cpu']))
    with tempfile.TemporaryDirectory(prefix='ice_workflow_') as temporary:
        directory = Path(temporary)
        cfg = directory / 'experiment.json'
        cfg.write_text(json.dumps(experiment.values), encoding='utf-8')
        out = directory / 'out'
        args = ['--config', str(cfg), '--output', str(out), '--workers', '1']
        completed = 0

        def interrupted(*values, **keywords):
            nonlocal completed
            if completed == 2:
                raise RuntimeError('Test interruption after two saved CQ points')
            result = evaluate_survey(*values, **keywords)
            completed += 1
            return result

        with patch('ice_imaging_2d.ice2d.forward_cq.adapter.evaluate_survey', side_effect=interrupted):
            with contextlib.redirect_stderr(io.StringIO()):
                reject(lambda: main(['-forward', *args]))
        assert not (out / 'data/flat/data_cq.npz').exists()
        with patch('ice_imaging_2d.ice2d.bottom_stage.invert', side_effect=AssertionError('forward called inversion')):
            main(['-forward', *args])
        path = out / 'data/flat/data_cq.npz'
        assert path.is_file() and not (out / 'inversion').exists()
        forward = experiment.for_case('flat')
        data = read_observation(path, forward)
        assert json.loads(str(data['metadata_json']))['cached_points'] == 2
        assert data['pressure'].shape[:2] == (2, 3)
        with patch('ice_imaging_2d.ice2d.forward_cq.adapter.create_model', side_effect=AssertionError('cache called solver')):
            cached, metadata = simulate(forward, checkpoint_dir=path.parent / 'cq_cache')
        assert metadata['cached_points'] == len(forward.grid().selected)
        assert np.array_equal(cached['pressure'], data['pressure'])
        # GPU execution choices do not change data identity. In a fresh process,
        # prohibit every PDE implementation import, then run inversion on CPU.
        experiment.values['forward']['solver']['backend'] = 'gpu'
        cfg.write_text(json.dumps(experiment.values), encoding='utf-8')
        script = '''
import importlib.abc, sys
class BlockPDE(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('adapter', 'coupled', 'water', 'ice', 'gpu', 'receiver', 'condensed')
        if fullname.startswith('ice_imaging_2d.ice2d.forward_cq.') and any(
            fullname.split('.')[3] == name for name in forbidden):
            raise AssertionError('Inversion attempted PDE import: ' + fullname)
sys.meta_path.insert(0, BlockPDE())
from ice_imaging_2d.main import main
main(sys.argv[1:])
'''
        command = [sys.executable, '-c', script, '-inv', '--config', str(cfg),
                   '--data', str(path), '--output', str(out)]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        runs = list((out / 'inversion/flat').iterdir())
        assert len(runs) == 1 and (runs[0] / 'images.npz').is_file()
        experiment.values['imaging']['image_x_step'] = .025
        cfg.write_text(json.dumps(experiment.values), encoding='utf-8')
        with patch('ice_imaging_2d.ice2d.forward_cq.adapter.simulate', side_effect=AssertionError('inversion called solver')):
            main(['-inv', '--config', str(cfg), '--data', str(path), '--output', str(out)])
        assert len(list((out / 'inversion/flat').iterdir())) == 2
        changed = deepcopy(forward.values)
        changed['materials']['water_speed'] += 1
        from ..ice2d.forward_cq.config import ForwardConfig
        reject(lambda: read_observation(path, ForwardConfig.from_dict(changed)))
        missing = directory / 'missing'
        with contextlib.redirect_stderr(io.StringIO()):
            reject(lambda: main(['-inv', '--config', str(cfg), '--output', str(missing)]))
        assert not missing.exists()
        # Default mode reuses complete data and then runs inversion.
        experiment.values['forward']['solver']['backend'] = 'cpu'
        cfg.write_text(json.dumps(experiment.values), encoding='utf-8')
        with patch('ice_imaging_2d.ice2d.forward_cq.adapter.simulate', side_effect=AssertionError('complete data recomputed')):
            main(args)
    return dict(passed=True, forward_only=True, inverse_with_blocked_pde_imports=True,
                gpu_config_cpu_inversion=True, interrupted_cq_checkpoint_resume=True,
                independent_imaging_runs=True, incompatible_data_rejected=True,
                default_two_stages=True, neural_network='skipped')


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
