"""Scientific figures. English plot labels avoid platform-specific CJK fonts."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from .model import ice_bottom
from .metrics import extract_ridge


plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 170})


def save_figure(fig, output, name):
    path = Path(output) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def model_and_pulse(data, config, case, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    x = np.linspace(-config.horizontal_halfwidth, config.horizontal_halfwidth, 1000)
    bottom = ice_bottom(x, config, case)
    axes[0].fill_between(x, -config.seabed_depth, bottom, color="#b8dfed", label="Water")
    if case == "soft":
        axes[0].fill_between(x, bottom, config.ice_top, color="#eeeeee", label="Pressure-release exterior")
    else:
        axes[0].fill_between(x, bottom, config.ice_top, color="#5385a0", label="Elastic ice")
    axes[0].plot(x, bottom, color="#163e50", lw=1)
    axes[0].scatter(data["receiver_x"], np.full_like(data["receiver_x"], float(data["array_z"])), s=5, color="#244660")
    axes[0].scatter(data["source_x"], np.full_like(data["source_x"], float(data["array_z"])), marker="*", s=65, color="#ca6236", label="Sources")
    axes[0].axhline(-config.seabed_depth, color="black", lw=2)
    axes[0].set(xlabel="x (m)", ylabel="z (m)", title=f"{case}: model and acquisition")
    axes[0].legend(loc="lower right", fontsize=8)
    times = np.linspace(0, 0.0006, 1500)
    pulse = np.cos(2 * np.pi * config.source_frequency * (times - config.source_center))
    pulse *= np.exp(-0.5 * ((times - config.source_center) / config.source_sigma) ** 2)
    axes[1].plot(times * 1e3, pulse, color="#244660")
    axes[1].set(xlabel="Time (ms)", ylabel="Relative stress-rate source", title="Gaussian-modulated source")
    frequency = np.linspace(0, 35000, 600)
    w = 2 * np.pi * frequency
    source_spectrum = np.exp(-0.5 * config.source_sigma ** 2 * (w - 2 * np.pi * config.source_frequency) ** 2)
    axes[2].plot(frequency / 1000, source_spectrum, color="#244660")
    axes[2].axvspan(config.band_low / 1000, config.band_high / 1000, color="#b8dfed", alpha=0.6)
    axes[2].set(xlabel="Frequency (kHz)", ylabel="Relative source amplitude", title="Processing band is 10-20 kHz")
    return save_figure(fig, output, f"{case}_model")


def gather_plot(data, background, processed, config, case, output):
    s = len(data["source_x"]) // 2
    r = int(np.argmin(abs(data["receiver_x"] - data["source_x"][s])))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    extent = [data["receiver_x"][0], data["receiver_x"][-1], data["time"][-1] * 1e3, data["time"][0] * 1e3]
    for ax, values, label in ((axes[0], data["pressure"][s], "Total pressure"),
                             (axes[1], processed["pressure"][s], "Background-subtracted, bandpassed")):
        clip = np.quantile(abs(values), 0.985)
        ax.imshow(values.T, extent=extent, aspect="auto", cmap="RdBu_r", vmin=-clip, vmax=clip)
        ax.set(xlabel="Receiver x (m)", ylabel="Time (ms)", title=label)
    arrival = processed["reference_arrival"][s] * 1e3
    axes[1].plot(data["receiver_x"], arrival - config.first_window_halfwidth * 1e3, "k--", lw=0.8)
    axes[1].plot(data["receiver_x"], arrival + config.first_window_halfwidth * 1e3, "k--", lw=0.8)
    trace = processed["pressure"][s, r]
    axes[2].plot(data["time"] * 1e3, trace, lw=0.8, color="#345570", label="Scattered pressure")
    axes[2].plot(data["time"] * 1e3, abs(hilbert(trace)), color="#cb6848", lw=1, label="Envelope")
    tb = config.source_center + 2 * (config.mean_bottom - float(data["array_z"])) / config.water_speed
    ta = tb + 2 * (config.ice_top - config.mean_bottom) / config.ice_p_speed
    for value, label in ((tb, "Ice bottom"), (ta, "P top echo")):
        axes[2].axvline(value * 1e3, color="#777777", ls="--", lw=0.8)
        axes[2].text(value * 1e3, max(trace) * 0.85, label, rotation=90, va="top", fontsize=8)
    axes[2].set(xlabel="Time (ms)", ylabel="Pressure (arbitrary source amplitude)", title="Near-zero-offset trace")
    axes[2].legend(loc="lower left", fontsize=8)
    return save_figure(fig, output, f"{case}_records")


def inversion_plot(result, config, case, output):
    truth = ice_bottom(result["x"], config, case)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    extent = [result["x"][0], result["x"][-1], result["z"][0], result["z"][-1]]
    for ax, name, label in ((axes[0], "migration", "Envelope delay-and-sum"),
                            (axes[1], "coherent", "Coherent delay-and-sum")):
        image = result[name]
        im = ax.imshow(image / max(float(image.max()), 1e-30), extent=extent, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        ridge, _ = extract_ridge(image, result["z"])
        ax.plot(result["x"], truth, color="#26d9d0", lw=1.5, label="True ice bottom")
        ax.plot(result["x"], ridge, color="white", ls="--", lw=1, label="Extracted ridge")
        ax.set(xlabel="x (m)", ylabel="z (m)", title=label)
    fig.colorbar(im, ax=axes[:2], shrink=0.75, label="Normalized image amplitude")
    for name, label, color in (("migration", "Envelope", "#d97542"), ("coherent", "Coherent", "#537ca1")):
        ridge, _ = extract_ridge(result[name], result["z"])
        axes[2].plot(result["x"], ridge, label=label, color=color)
    axes[2].plot(result["x"], truth, "k--", lw=1.8, label="Truth")
    axes[2].axhline(config.mean_bottom, color="#aaaaaa", ls=":", label="Constant estimate")
    axes[2].set(xlabel="x (m)", ylabel="Ice bottom z (m)", title=f"{case}: recovered profile")
    axes[2].legend(fontsize=8)
    return save_figure(fig, output, f"{case}_inversion")


def dsm_plot(result, config, case, output):
    if "dsm" not in result:
        return None
    truth = ice_bottom(result["x"], config, case)
    names = ["dsm", "dsm_corrected", "correction_only"]
    titles = ["Cauchy backpropagation energy", "With synthetic mirror correction", "Correction alone (zero data)"]
    scale = max(float(result[name].max()) for name in names)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    extent = [result["x"][0], result["x"][-1], result["z"][0], result["z"][-1]]
    for ax, name, title in zip(axes, names, titles):
        values = result[name] / max(scale, 1e-30)
        im = ax.imshow(values, extent=extent, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ridge, _ = extract_ridge(result[name], result["z"])
        ax.plot(result["x"], truth, color="#ff7043", lw=1.5)
        ax.plot(result["x"], ridge, "w--", lw=1)
        ax.set(xlabel="x (m)", ylabel="z (m)", title=title)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Energy / shared maximum")
    path = save_figure(fig, output, f"{case}_dsm")
    if "dsm_impedance" in result:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, name, title in ((axes[0], "dsm_corrected", "Original mirror correction"),
                                (axes[1], "dsm_impedance", "Normal-incidence impedance correction")):
            values = result[name]
            ax.imshow(values / max(float(values.max()), 1e-30), extent=extent, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ridge, _ = extract_ridge(values, result["z"])
            ax.plot(result["x"], truth, color="#ff7043", lw=1.5, label="Truth")
            ax.plot(result["x"], ridge, "w--", lw=1, label="Extracted ridge")
            ax.set(xlabel="x (m)", ylabel="z (m)", title=title)
        save_figure(fig, output, f"{case}_dsm_impedance")
    return path


def snapshot_plot(data, config, case, output):
    if len(data["snapshots"]) == 0:
        return None
    fig, axes = plt.subplots(2, len(data["snapshots"]), figsize=(12, 7), squeeze=False, constrained_layout=True)
    extent = [data["snapshot_x"][0], data["snapshot_x"][-1], data["snapshot_z"][-1], data["snapshot_z"][0]]
    for column, snapshot in enumerate(data["snapshots"]):
        for row, label in ((0, "Mean normal stress"), (1, "Velocity curl")):
            values = abs(snapshot[row])
            maximum = max(float(values.max()), 1e-30)
            db = 20 * np.log10(np.maximum(values / maximum, 1e-4))
            im = axes[row, column].imshow(db, extent=extent, aspect="auto", origin="upper", cmap="inferno", vmin=-60, vmax=0)
            axes[row, column].plot(data["snapshot_x"], ice_bottom(data["snapshot_x"], config, case), color="#65d7d1", lw=0.8)
            axes[row, column].set(xlabel="x (m)", ylabel="z (m)", ylim=(-4.4, 0.2), title=f"{label}, {data['snapshot_times'][column] * 1000:.2f} ms")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Amplitude (dB relative to each panel maximum)")
    return save_figure(fig, output, f"{case}_wavefields")


def robustness_plot(rows, output):
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    locations = np.arange(len(rows))
    ax.barh(locations - 0.18, [100 * row["rmse_m"] for row in rows], height=0.32, color="#416b88", label="Envelope")
    if all("coherent_rmse_m" in row for row in rows):
        ax.barh(locations + 0.18, [100 * row["coherent_rmse_m"] for row in rows], height=0.32, color="#c6734b", label="Coherent")
    ax.set_yticks(locations, [row["experiment"] for row in rows])
    ax.axvline(5, color="#b95c40", ls="--", lw=1, label="5 cm working target")
    ax.invert_yaxis()
    ax.set(xlabel="Ice-bottom RMSE (cm)", title="Sensitivity on the same full-wave sine data")
    ax.legend()
    return save_figure(fig, output, "robustness")

