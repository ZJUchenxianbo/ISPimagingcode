"""Second/fourth-order staggered spatial differences, second order in time.

The positive downward computational coordinate is converted back to physical z
in the recorded normal derivative. Fluids have mu=0. Vacuum is represented by
zero mass/stiffness outside the material, a low-order stair-step free surface.
The exact interface location and this approximation are checked by refinement.
"""

import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit

from .model import make_model, acquisition_metadata


SOLVER_VERSION = "staggered-air-order2-smooth-linear-v3"


@njit(cache=True)
def _advance(lam, mu, corner_mu, bx, bz, damp, dt, h, source, source_nodes, source_weights,
             receiver_nodes, receiver_weights, stride, snapshot_steps, spatial_order=4, progress_label=""):
    nz, nx = lam.shape
    vx = np.zeros((nz, nx), np.float32)
    vz = np.zeros((nz, nx), np.float32)
    sxx = np.zeros((nz, nx), np.float32)
    szz = np.zeros((nz, nx), np.float32)
    sxz = np.zeros((nz, nx), np.float32)
    samples = (len(source) - 1) // stride + 1
    pressure = np.zeros((len(receiver_nodes), samples), np.float32)
    gradient = np.zeros_like(pressure)
    snapshots = np.zeros((len(snapshot_steps), 2, (nz + 2) // 3, (nx + 2) // 3), np.float32)
    factor = dt / h
    # A nearest-neighbor stencil avoids fourth-order links that jump across
    # the air/solid density discontinuity. Keep the legacy fourth-order option.
    a = 1.0 if spatial_order == 2 else 9.0 / 8.0
    b = 0.0 if spatial_order == 2 else -1.0 / 24.0
    bottom = nz - 3
    snapshot_index = 0
    progress_stride = max(1, len(source) // 4)
    for it in range(len(source)):
        if progress_label and it % progress_stride == 0:
            print(progress_label, "time steps:", it, "/", len(source))
        # Rigid horizontal seabed at a staggered vertical-velocity face.
        for ix in range(nx):
            sxx[bottom + 1, ix] = sxx[bottom, ix]
            sxx[bottom + 2, ix] = sxx[bottom - 1, ix]
            szz[bottom + 1, ix] = szz[bottom, ix]
            szz[bottom + 2, ix] = szz[bottom - 1, ix]
        for iz in range(2, nz - 2):
            for ix in range(2, nx - 2):
                dx_sxx = a * (sxx[iz, ix + 1] - sxx[iz, ix]) + b * (sxx[iz, ix + 2] - sxx[iz, ix - 1])
                dz_sxz = a * (sxz[iz, ix] - sxz[iz - 1, ix]) + b * (sxz[iz + 1, ix] - sxz[iz - 2, ix])
                dx_sxz = a * (sxz[iz, ix] - sxz[iz, ix - 1]) + b * (sxz[iz, ix + 1] - sxz[iz, ix - 2])
                dz_szz = a * (szz[iz + 1, ix] - szz[iz, ix]) + b * (szz[iz + 2, ix] - szz[iz - 1, ix])
                vx[iz, ix] = damp[iz, ix] * (vx[iz, ix] + factor * bx[iz, ix] * (dx_sxx + dz_sxz))
                vz[iz, ix] = damp[iz, ix] * (vz[iz, ix] + factor * bz[iz, ix] * (dx_sxz + dz_szz))
        for ix in range(nx):
            vz[bottom, ix] = 0
            vz[bottom + 1, ix] = -vz[bottom - 1, ix]
            vz[bottom + 2, ix] = -vz[bottom - 2, ix]
            vx[bottom + 1, ix] = vx[bottom, ix]
            vx[bottom + 2, ix] = vx[bottom - 1, ix]
        for iz in range(2, nz - 2):
            for ix in range(2, nx - 2):
                dx_vx = a * (vx[iz, ix] - vx[iz, ix - 1]) + b * (vx[iz, ix + 1] - vx[iz, ix - 2])
                dz_vz = a * (vz[iz, ix] - vz[iz - 1, ix]) + b * (vz[iz + 1, ix] - vz[iz - 2, ix])
                dz_vx = a * (vx[iz + 1, ix] - vx[iz, ix]) + b * (vx[iz + 2, ix] - vx[iz - 1, ix])
                dx_vz = a * (vz[iz, ix + 1] - vz[iz, ix]) + b * (vz[iz, ix + 2] - vz[iz, ix - 1])
                longitudinal = lam[iz, ix] + 2 * mu[iz, ix]
                sxx[iz, ix] = damp[iz, ix] * (sxx[iz, ix] + factor * (longitudinal * dx_vx + lam[iz, ix] * dz_vz))
                szz[iz, ix] = damp[iz, ix] * (szz[iz, ix] + factor * (lam[iz, ix] * dx_vx + longitudinal * dz_vz))
                sxz[iz, ix] = damp[iz, ix] * (sxz[iz, ix] + factor * corner_mu[iz, ix] * (dz_vx + dx_vz))
        for q in range(len(source_weights)):
            iz, ix = source_nodes[q]
            amplitude = dt * source[it] * source_weights[q]
            sxx[iz, ix] -= amplitude
            szz[iz, ix] -= amplitude
        if it % stride == 0:
            k = it // stride
            for r in range(len(receiver_nodes)):
                for q in range(receiver_weights.shape[1]):
                    iz, ix = receiver_nodes[r, q]
                    weight = receiver_weights[r, q]
                    pressure[r, k] += weight * -0.5 * (sxx[iz, ix] + szz[iz, ix])
                    # Physical +z is opposite the increasing array row direction.
                    p_above = -0.5 * (sxx[iz - 1, ix] + szz[iz - 1, ix])
                    p_below = -0.5 * (sxx[iz + 1, ix] + szz[iz + 1, ix])
                    gradient[r, k] += weight * (p_above - p_below) / (2 * h)
        if snapshot_index < len(snapshot_steps) and it == snapshot_steps[snapshot_index]:
            for iz in range(3, nz - 3, 3):
                for ix in range(3, nx - 3, 3):
                    snapshots[snapshot_index, 0, iz // 3, ix // 3] = -0.5 * (sxx[iz, ix] + szz[iz, ix])
                    # Both derivatives are evaluated at the same staggered
                    # shear-stress corner, so an acoustic gradient has zero curl.
                    curl_x = a * (vz[iz, ix + 1] - vz[iz, ix]) + b * (vz[iz, ix + 2] - vz[iz, ix - 1])
                    curl_z = a * (vx[iz + 1, ix] - vx[iz, ix]) + b * (vx[iz + 2, ix] - vx[iz - 1, ix])
                    snapshots[snapshot_index, 1, iz // 3, ix // 3] = (curl_x - curl_z) / h
            snapshot_index += 1
    return pressure, gradient, snapshots


def _one_shot(task):
    config, case, shot, cache_dir, geometry, quiet = task
    label = f"Forward {case} shot {shot + 1}/{config.source_count}"
    if not quiet:
        print(f"{label}: checking cache", flush=True)
    key = hashlib.sha256(json.dumps([SOLVER_VERSION, config.as_dict(), case, shot, None if geometry is None else geometry.as_dict()], sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{case}_shot{shot:02d}_{key}.npz"
    if path.exists():
        if not quiet:
            print(f"{label}: using cached shot", flush=True)
        with np.load(path, allow_pickle=False) as old:
            return {name: old[name] for name in old.files}
    model = make_model(config, case, geometry)
    times = (np.arange(model["steps"]) + 0.5) * model["dt"]
    pulse = np.exp(-0.5 * ((times - config.source_center) / config.source_sigma) ** 2)
    pulse *= np.cos(2 * np.pi * config.source_frequency * (times - config.source_center))
    # Fix integrated spatial source strength under mesh refinement.
    source = (pulse * 1e4 / config.grid_spacing ** 2).astype(np.float32)
    snapshot_steps = np.empty(0, dtype=np.int64)
    if config.snapshots and shot == config.source_count // 2 and case != "background":
        target_times = np.array([0.0015, 0.0024, 0.0032])
        snapshot_steps = np.rint(target_times[target_times < config.record_seconds] / model["dt"]).astype(np.int64)
    if not quiet:
        print(f"{label}: grid {model['lam'].shape}, {model['steps']} time steps; first call may compile Numba", flush=True)
    started = time.perf_counter()
    pressure, gradient, snapshots = _advance(
        model["lam"], model["mu"], model["corner_mu"], model["inverse_density_x"],
        model["inverse_density_z"], model["damping"], model["dt"], config.grid_spacing,
        source, model["source_nodes"][shot], model["source_weights"][shot],
        model["receiver_nodes"], model["receiver_weights"], model["stride"], snapshot_steps, config.spatial_order, "" if quiet else label)
    if not np.isfinite(pressure).all() or not np.isfinite(gradient).all():
        raise FloatingPointError("The forward run contains non-finite fields.")
    result = dict(pressure=pressure, gradient=gradient, snapshots=snapshots,
                  snapshot_times=snapshot_steps * model["dt"],
                  snapshot_x=model["x"][::3], snapshot_z=model["z"][::3],
                  seconds=np.array(time.perf_counter() - started))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    if not quiet:
        print(f"{label}: completed in {float(result['seconds']):.2f} s", flush=True)
    return result


def simulate(config, case, output, workers=1, geometry=None, save=True, quiet=False):
    config.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output)
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tasks = [(config, case, shot, str(cache), geometry, quiet) for shot in range(config.source_count)]
    if workers == 1:
        records = []
        for task in tasks:
            if not quiet:
                print(f"Forward {case}: shot {task[2] + 1}/{config.source_count}", flush=True)
            records.append(_one_shot(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = []
            for shot, record in enumerate(pool.map(_one_shot, tasks)):
                records.append(record)
                if not quiet:
                    print(f"Forward {case}: shot {shot + 1}/{config.source_count} complete", flush=True)
    model = acquisition_metadata(config)
    centre = records[len(records) // 2]
    result = dict(pressure=np.stack([item["pressure"] for item in records]),
                  gradient=np.stack([item["gradient"] for item in records]),
                  time=(np.arange(records[0]["pressure"].shape[-1]) * model["stride"] + 1) * model["dt"],
                  source_x=model["source_x"], receiver_x=model["receiver_x"], array_z=np.array(model["array_z"]),
                  snapshots=centre["snapshots"], snapshot_times=centre["snapshot_times"],
                  snapshot_x=centre["snapshot_x"], snapshot_z=centre["snapshot_z"],
                  solver_seconds=np.array(sum(float(item["seconds"]) for item in records)),
                  config_json=np.array(json.dumps(config.as_dict(), sort_keys=True)),
                  case=np.array(case), solver_version=np.array(SOLVER_VERSION),
                  geometry_json=np.array(json.dumps(None if geometry is None else geometry.as_dict(), sort_keys=True)),
                  source_nodes=model["source_nodes"], source_weights=model["source_weights"],
                  receiver_nodes=model["receiver_nodes"], receiver_weights=model["receiver_weights"])
    if save:
        np.savez_compressed(output / f"data_{case}.npz", **result)
    return result
