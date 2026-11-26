"""Shared numerical function extracted from the supplied solver."""
import numpy as np

def flat_compliance(s, k, thickness, *, ice_density=900.0, ice_vp=3600.0, ice_vs=1800.0,
                    air_loading=True, air_speed=340.3, air_density=1.225):
    """Return upward bottom displacement per unit pressure on the ice bottom."""
    k = np.asarray(k, dtype=float)
    shear = ice_density * ice_vs**2
    qp = np.sqrt(k**2 + (s / ice_vp)**2)
    qs = np.sqrt(k**2 + (s / ice_vs)**2)
    vertical = np.stack((qp, qp, qs, qs), axis=-1)
    derivative = vertical * np.array([-1.0, 1.0, -1.0, 1.0])
    tail = np.exp(-vertical * thickness)
    ones = np.ones_like(k)
    bottom = np.stack((ones, tail[..., 1], ones, tail[..., 3]), axis=-1)
    top = np.stack((tail[..., 0], ones, tail[..., 2], ones), axis=-1)

    ik = 1j * k[..., None]
    displacement = np.concatenate((derivative[..., :2], ik * np.ones(2)), axis=-1)
    shear_traction = np.concatenate((2.0 * shear * ik * derivative[..., :2],
                                     -shear * (qs**2 + k**2)[..., None] * np.ones(2)), axis=-1)
    pressure_term = np.asarray(ice_density * s**2 + 2.0 * shear * k**2)
    normal_traction = np.concatenate((pressure_term[..., None] * np.ones(2),
                                      2.0 * shear * ik * derivative[..., 2:]), axis=-1)
    top_traction = normal_traction
    if air_loading:
        qa = np.sqrt(k**2 + (s / air_speed)**2)
        # p_a(0) = rho_a s^2 u_z / q_a for p_a(z) = p_a(0) exp(-q_a z).
        top_traction = top_traction + (air_density * s**2 / qa)[..., None] * displacement
    matrix = np.stack((shear_traction * bottom, normal_traction * bottom,
                       shear_traction * top, top_traction * top), axis=-2)

    scale = shear * (k**2 + abs(s / ice_vs)**2)
    rhs = np.zeros(k.shape + (4, 1), dtype=complex)
    rhs[..., 1, 0] = -1.0
    amplitudes = np.linalg.solve(matrix / scale[..., None, None], rhs)[..., 0] / scale[..., None]
    return np.sum(amplitudes * displacement * bottom, axis=-1)
