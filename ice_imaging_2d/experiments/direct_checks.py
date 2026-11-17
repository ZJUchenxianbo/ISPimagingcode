"""Small operator checks against explicit sums, without a forward solve."""
import json
import tempfile
from pathlib import Path
import numpy as np
from ..ice2d.background_green import PressureGreen
from ..ice2d.direct_imaging import cauchy_backpropagate
from ..ice2d.geometry import line_weights


def main():
    rng = np.random.default_rng(29)
    time = .02 + np.arange(7) * .001
    lag = np.arange(4) * .001
    receivers = np.array([-1., -.2, .6])
    metadata = dict(schema='ice-cauchy-green-v1', response='causal_impulse',
                    normal='positive_z', background='air-elastic_ice-water',
                    derivative='receiver_normal', sample_medium='water',
                    source_equation='pressure_rate_rhs', reference_signature='synthetic',
                    validation_report='algebra-only test, not physical Green validation')
    # Complex kernels test the conjugation in the adjoint, not just real reversal.
    g = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    gn = rng.normal(size=g.shape) + 1j * rng.normal(size=g.shape)
    kernel = PressureGreen(g, gn, lag, np.array([[0., -2.], [.2, -2.]]), receivers, -3., metadata)
    p, pn = rng.normal(size=(2, 3, 7)), rng.normal(size=(2, 3, 7))
    result = cauchy_backpropagate(p, pn, time, kernel)
    direct = np.zeros_like(result['field'])
    for source in range(2):
        for point in range(2):
            for index, shift in enumerate(range(-3, 7)):
                for receiver, wr in enumerate(line_weights(receivers)):
                    for tau, wt in enumerate(line_weights(time)):
                        k = tau - shift
                        if 0 <= k < len(lag):
                            direct[source, point, index] += wr * wt * (pn[source, receiver, tau] * g[point, receiver, k].conjugate() - p[source, receiver, tau] * gn[point, receiver, k].conjugate())
    error = float(np.max(abs(result['field'] - direct)))
    assert np.allclose(result['field'], direct, atol=1e-14, rtol=1e-12)
    zero = cauchy_backpropagate(np.zeros_like(p), np.zeros_like(p), time, kernel)
    assert np.all(zero['field'] == 0)
    # Changing both normal definitions reverses the whole boundary pairing.
    opposite = PressureGreen(g, -gn, lag, kernel.points, receivers, -3., metadata)
    flipped = cauchy_backpropagate(p, -pn, time, opposite)
    assert np.allclose(flipped['field'], -result['field'], atol=1e-14)
    # A tail pulse and delayed impulse must not wrap around as circular FFT data.
    impulse_kernel = PressureGreen(np.ones((1, 1, 4)), np.zeros((1, 1, 4)), lag,
                                   np.array([[0., -2.]]), np.array([0.]), -3., metadata)
    impulse = np.zeros((1, 1, 7)); impulse[0, 0, -1] = 1
    tail = cauchy_backpropagate(np.zeros_like(impulse), impulse, time, impulse_kernel)
    assert np.allclose(tail['field'][0, 0, :6], 0, atol=1e-14)
    try:
        cauchy_backpropagate(p, pn, time * 2, kernel)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched time grids accepted')
    # Exercise disk contract and CLI orchestration with synthetic arrays only.
    from argparse import Namespace
    from ..ice2d import direct_workflow as direct_experiment
    from ..ice2d.background_green import reference_signature
    with tempfile.TemporaryDirectory(prefix='ice_direct_check_') as directory:
        path = Path(directory)
        reference = dict(pressure=np.zeros_like(p), gradient=np.zeros_like(p), time=time,
                         source_x=np.array([-.5,.5]), receiver_x=receivers, array_z=np.array(-3.),
                         config_json=np.array(json.dumps(dict(test_only=True))),
                         geometry_json=np.array(json.dumps(dict(top=0., nodes=[-2.,2.], bottom=[-1.,-1.]))),
                         solver_version=np.array('synthetic-algebra-only'))
        observed = dict(reference, pressure=p, gradient=pn)
        metadata['reference_signature'] = reference_signature(reference)
        np.savez(path / 'observed.npz', **observed)
        np.savez(path / 'reference.npz', **reference)
        np.savez(path / 'green.npz', gpp=g, normal_gpp=gn, lag=lag,
                 points=kernel.points, receiver_x=receivers, array_z=-3.,
                 metadata_json=json.dumps(metadata))
        args = Namespace(stage='backpropagate', estimate=False,
                         observed=str(path / 'observed.npz'), reference_data=str(path / 'reference.npz'),
                         green=str(path / 'green.npz'), output=str(path))
        direct_experiment.run_direct(args)
        with np.load(path / 'direct_backpropagation/backpropagated_field.npz') as saved:
            assert np.allclose(saved['field'], direct, rtol=1e-12, atol=1e-14)
        meta = json.loads((path / 'direct_backpropagation/metadata.json').read_text())
        assert not meta['output_is_interface_indicator'] and meta['inversion_forward_calls'] == 0
        observed.pop('gradient')
        np.savez(path / 'observed.npz', **observed)
        try:
            direct_experiment.run_direct(args)
        except ValueError as failure:
            assert 'Pressure-only' in str(failure)
        else:
            raise AssertionError('Missing normal data was silently accepted')
    print(json.dumps(dict(passed=True, explicit_sum_max_error=error,
                         conjugation=True, normal_sign=True, no_wrap=True,
                         zero_residual=True, physical_validation=False), indent=2))


if __name__ == '__main__':
    main()
