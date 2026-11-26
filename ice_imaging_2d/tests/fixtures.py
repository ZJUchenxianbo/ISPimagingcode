"""Private coarse numerical fixtures; main never selects these test grids."""

import json
from pathlib import Path

from ..ice2d.config import ExperimentConfig, ImagingConfig
from ..ice2d.forward_cq.config import ForwardConfig


def coarse_forward():
    return ForwardConfig.from_dict(dict(
        geometry=dict(width_m=2., water_depth_m=2.5, pml_x_m=.5, pml_z_m=.5,
                      top='0', bottom='-0.7 + 0.03*sin(pi*x)'),
        acquisition=dict(sources=[[-.3, -1.5], [.3, -1.5]],
                         receivers=[[-.6, -1.5], [0, -1.5], [.6, -1.5]]),
        frequency=dict(carrier_hz=500., min_hz=100., max_hz=900.,
                       duration_s=.03, dt_s=.0001, sigma_per_s=160.),
        pulse=dict(kind='gaussian', sigma_s=.0025, center_s=.012),
        processing=dict(band_low=100., band_high=900.),
        solver=dict(backend='cpu', water_panel_m=.5)))


def coarse_imaging():
    return ImagingConfig(water_speed=1500., ice_p_speed=3600., ice_top=0.,
                         source_center=.012, band_low=100., band_high=900.,
                         mean_bottom=-.7, image_halfwidth=.1, image_x_step=.1,
                         image_z_min=-.9, image_z_max=-.5, image_z_step=.1,
                         first_window_halfwidth=.004, maximum_offset=1.)


def small_experiment():
    """Formal physical units/band, deliberately coarse mesh for workflow tests."""
    path = Path(__file__).resolve().parents[1] / 'configs/experiment.json'
    values = json.loads(path.read_text(encoding='utf-8'))
    forward = values['forward']
    forward['geometry'].update(width_m=.4, water_depth_m=2., pml_x_m=.2, pml_z_m=.2)
    forward['acquisition'] = dict(sources=[[-.04, -1.7], [.04, -1.7]],
                                  receivers=[[-.06, -1.7], [0., -1.7], [.06, -1.7]])
    forward['frequency'].update(duration_s=.0015, sigma_per_s=4.8 / .0015)
    forward['solver'].update(backend='cpu', water_panel_m=.2, ice_elements_per_wavelength=1.)
    values['cases'] = dict(flat=dict(kind='flat', mean_thickness_m=1.5))
    values['imaging'].update(image_halfwidth=.05, image_x_step=.05,
                             image_z_min=-1.6, image_z_max=-1.4, image_z_step=.1,
                             first_window_halfwidth=.00013, maximum_offset=.2,
                             window_factors=[1.])
    return ExperimentConfig.from_dict(values)
