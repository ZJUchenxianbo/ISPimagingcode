"""Validated-input Cauchy backpropagation, not a completed dual indicator."""
import json
from pathlib import Path
import numpy as np
from .background_green import load_pressure_green, reference_signature
from .direct_imaging import cauchy_backpropagate
from .data_store import write_manifest

STATUS = dict(
    method='research proposal section 4.3: Cauchy adjoint backpropagation',
    complete_dual_interface_method=False,
    implemented=['Cauchy trace subtraction', 'receiver/time quadrature',
                 'adjoint time reversal', 'fixed-kernel backpropagation'],
    missing=['validated coupled Gpp/Gup/air/traction kernel construction',
             'section 4.4 coupled compensation field',
             'section 4.5 upper-interface matching operator and focusing validation',
             'local shape-kernel derivation if section 4.6 is chosen'],
    inversion_forward_calls=0,
)


def run_direct(args):
    if args.stage == 'status' or args.estimate:
        print(json.dumps(STATUS, ensure_ascii=False, indent=2))
        return
    if args.stage != 'backpropagate':
        raise ValueError('Direct mode supports status/backpropagate; sensitivity is an explicit comparison mode')
    if not all((args.observed, args.reference_data, args.green)):
        raise ValueError('backpropagate needs --observed, --reference-data and --green; kernels are not generated automatically')
    def load(path):
        with np.load(path, allow_pickle=False) as values:
            return {key: values[key] for key in values.files}
    observed, reference = load(args.observed), load(args.reference_data)
    for key in ('time', 'source_x', 'receiver_x', 'array_z', 'solver_version', 'config_json'):
        if not np.array_equal(observed[key], reference[key]):
            raise ValueError(f'Observation/reference mismatch: {key}')
    if 'gradient' not in observed or 'gradient' not in reference:
        raise ValueError('Pressure-only observations cannot silently supply normal Cauchy data')
    shape = (len(observed['source_x']), len(observed['receiver_x']), len(observed['time']))
    for dataset in (observed, reference):
        for name in ('pressure', 'gradient'):
            if dataset[name].shape != shape:
                raise ValueError(f'{name} must match source/receiver/time coordinate dimensions')
    kernel = load_pressure_green(args.green)
    if kernel.metadata['reference_signature'] != reference_signature(reference):
        raise ValueError('Kernel background differs from supplied reference')
    if not np.array_equal(kernel.receiver_x, observed['receiver_x']) or kernel.array_z != float(observed['array_z']):
        raise ValueError('Kernel acquisition differs from observed acquisition')
    if np.any(kernel.points[:, 1] >= np.interp(kernel.points[:, 0],
              json.loads(str(reference['geometry_json']))['nodes'],
              json.loads(str(reference['geometry_json']))['bottom'])):
        raise ValueError('Gpp scalar operator currently supports water sampling points only')
    result = cauchy_backpropagate(observed['pressure'] - reference['pressure'],
                                 observed['gradient'] - reference['gradient'], observed['time'], kernel)
    output = Path(args.output) / 'direct_backpropagation'
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'backpropagated_field.npz', **result)
    write_manifest(output / 'metadata.json', dict(**STATUS, observed=str(Path(args.observed).resolve()),
                   reference=str(Path(args.reference_data).resolve()), kernel=kernel.metadata,
                   normal='positive_z', correction_applied=False, output_is_interface_indicator=False))
    print(f'Cauchy field saved: {output}; compensation and dual indicators remain unimplemented')
