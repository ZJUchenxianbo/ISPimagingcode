"""Material maps on a staggered Cartesian grid, with positive-up physical z."""

import numpy as np

from .geometry import sampling_stencil


def ice_bottom(x, config, case):
    x = np.asarray(x)
    if case in ("flat", "background"):
        return np.full_like(x, config.mean_bottom, dtype=float)
    if case in ("sine", "soft"):
        return config.mean_bottom + config.rough_amplitude * np.sin(2 * np.pi * x / config.rough_period)
    if case == "bump":
        return config.mean_bottom + config.rough_amplitude * np.exp(-((x - 0.25) / 0.45) ** 2)
    if case == "multiscale":
        profile = 0.65 * np.sin(2 * np.pi * x / config.rough_period)
        profile += 0.35 * np.sin(2 * np.pi * x / 0.8 + 0.4)
        return config.mean_bottom + config.rough_amplitude * profile
    raise ValueError(f"Unknown model case {case!r}")


def make_model(config, case, geometry=None):
    if geometry is not None:
        geometry.validate(config)
    h = config.grid_spacing
    nx = int(round(2 * config.horizontal_halfwidth / h))
    nz = int(round((config.computational_top + config.seabed_depth) / h))
    # Two ghost cells on every side. Stress coordinates are at cell centres.
    x = -config.horizontal_halfwidth + (np.arange(nx + 4) - 1.5) * h
    z = config.computational_top - (np.arange(nz + 4) - 1.5) * h
    shape = (len(z), len(x))
    density = np.full(shape, config.water_density, dtype=np.float32)
    p_speed = np.full(shape, config.water_speed, dtype=np.float32)
    s_speed = np.zeros(shape, dtype=np.float32)
    bottom = ice_bottom(x, config, case) if geometry is None else geometry.evaluate(x)
    ice_top = config.ice_top if geometry is None else geometry.top
    if case != "background":
        ice = (z[:, None] >= bottom[None, :]) & (z[:, None] < ice_top)
        density[ice] = config.ice_density
        p_speed[ice] = config.ice_p_speed
        s_speed[ice] = config.ice_s_speed
        vacuum = np.broadcast_to(z[:, None] >= ice_top, shape).copy()
        if case == "soft":
            # The zero-stiffness region gives a stair-step pressure-release surface.
            vacuum = z[:, None] >= bottom[None, :]
        if config.upper_medium == "air" and case != "soft":
            density[vacuum] = config.air_density
            p_speed[vacuum] = config.air_speed
        else:
            density[vacuum] = 0
            p_speed[vacuum] = 0
        s_speed[vacuum] = 0

    mu = density * s_speed ** 2
    lam = density * p_speed ** 2 - 2 * mu
    if config.interface_scheme == "smooth" and case not in ("background", "soft"):
        # Compact C1 material transition (not an exact fitted-interface scheme).
        # Arithmetic rho/lambda/mu mixtures keep local wave speeds bounded by
        # constituent speeds. Harmonic corner mu below enforces zero shear in
        # pure-fluid cells. Width shrinks with h and must be convergence-tested.
        width = config.interface_width_cells * h
        def transition(distance):
            u = np.clip(0.5 + distance / width, 0, 1)
            return u * u * (3 - 2 * u)
        above_bottom = transition(z[:, None] - bottom[None, :])
        below_top = transition(ice_top - z[:, None])
        ice_fraction = above_bottom * below_top
        water_fraction = 1 - above_bottom
        air_fraction = above_bottom * (1 - below_top)
        air_rho = config.air_density if config.upper_medium == "air" else 0.0
        density = (water_fraction * config.water_density + ice_fraction * config.ice_density
                   + air_fraction * air_rho).astype(np.float32)
        mu = (ice_fraction * config.ice_density * config.ice_s_speed ** 2).astype(np.float32)
        lam = (water_fraction * config.water_density * config.water_speed ** 2
               + ice_fraction * config.ice_density * (config.ice_p_speed ** 2 - 2 * config.ice_s_speed ** 2)
               + air_fraction * air_rho * config.air_speed ** 2).astype(np.float32)
    inverse_density_x = np.zeros(shape, dtype=np.float32)
    inverse_density_z = np.zeros(shape, dtype=np.float32)
    total_x = density[:, :-1] + density[:, 1:]
    total_z = density[:-1] + density[1:]
    np.divide(2, total_x, out=inverse_density_x[:, :-1], where=total_x > 0)
    np.divide(2, total_z, out=inverse_density_z[:-1], where=total_z > 0)
    corner_mu = np.zeros(shape, dtype=np.float32)
    corners = (mu[:-1, :-1], mu[1:, :-1], mu[:-1, 1:], mu[1:, 1:])
    all_solid = np.logical_and.reduce([part > 0 for part in corners])
    reciprocal_sum = np.zeros_like(corners[0])
    for part in corners:
        reciprocal_sum += np.divide(1, part, out=np.zeros_like(part), where=part > 0)
    np.divide(4, reciprocal_sum, out=corner_mu[:-1, :-1], where=all_solid)

    dt = config.time_step
    side = np.maximum((np.abs(x) - config.horizontal_halfwidth + config.sponge_width) / config.sponge_width, 0)
    top = np.maximum((z - config.computational_top + config.sponge_width) / config.sponge_width, 0)
    if case != "background" and (config.upper_medium == "free_surface" or case == "soft"):
        # The upper domain is vacuum. Only the background water needs an upper
        # absorber; a raised physical ice surface must never enter that absorber.
        top[:] = 0
    damping = np.exp(-dt * config.sponge_rate * (top[:, None] ** 3 + side[None, :] ** 3)).astype(np.float32)
    acquisition = acquisition_metadata(config, x, z)
    return dict(density=density, lam=lam, mu=mu, corner_mu=corner_mu,
                inverse_density_x=inverse_density_x, inverse_density_z=inverse_density_z,
                damping=damping,
                **acquisition)


def acquisition_metadata(config, x=None, z=None):
    """Small coordinate/stencil arrays; no material grids allocated."""
    h = config.grid_spacing
    if x is None:
        nx = round(2 * config.horizontal_halfwidth / h)
        nz = round((config.computational_top + config.seabed_depth) / h)
        x = -config.horizontal_halfwidth + (np.arange(nx + 4) - 1.5) * h
        z = config.computational_top - (np.arange(nz + 4) - 1.5) * h
    dt = config.time_step
    target_source_x = np.linspace(-config.source_halfwidth, config.source_halfwidth, config.source_count)
    if config.source_count == 1:
        target_source_x = np.array([0.0])
    count = int(np.floor(2 * config.receiver_halfwidth / config.receiver_spacing + 1e-9)) + 1
    target_receiver_x = -config.receiver_halfwidth + np.arange(count) * config.receiver_spacing
    if config.acquisition == "snap":
        # Preserve the legacy target sequence, including its tie-breaking behavior.
        target_receiver_x = np.arange(-config.receiver_halfwidth, config.receiver_halfwidth + h, config.receiver_spacing)
    source_indices = np.abs(x[:, None] - target_source_x).argmin(axis=0)
    receiver_indices = np.unique(np.abs(x[:, None] - target_receiver_x).argmin(axis=0))
    receiver_z_index = int(np.argmin(abs(z + config.array_depth)))
    if config.acquisition == "snap":
        target_receiver_x = x[receiver_indices]
    source_nodes, source_weights = sampling_stencil(x, z, target_source_x, config.array_depth, config.acquisition)
    receiver_nodes, receiver_weights = sampling_stencil(x, z, target_receiver_x, config.array_depth, config.acquisition)
    source_x = np.sum(x[source_nodes[:, :, 1]] * source_weights, axis=1)
    receiver_x = np.sum(x[receiver_nodes[:, :, 1]] * receiver_weights, axis=1)
    array_z = float(np.sum(z[receiver_nodes[0, :, 0]] * receiver_weights[0]))
    # Source is on the recorded pressure line. Subtract its homogeneous response
    # before imaging, including the coincident source/receiver sample.
    stride = max(1, int(round(config.output_interval / dt)))
    steps = int(np.ceil(config.record_seconds / dt))
    return dict(x=x, z=z, dt=dt, stride=stride, steps=steps,
                source_indices=source_indices, receiver_indices=receiver_indices,
                receiver_z_index=receiver_z_index, source_x=source_x,
                receiver_x=receiver_x, array_z=array_z, source_nodes=source_nodes,
                source_weights=source_weights, receiver_nodes=receiver_nodes,
                receiver_weights=receiver_weights)
