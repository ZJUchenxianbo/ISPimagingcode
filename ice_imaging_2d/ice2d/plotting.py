"""Pure plots of first-reflection data and TD-DSM indicators."""
import numpy as np
from .metrics import extract_ridge

METHODS = ("dsm", "dsm_corrected", "correction_only")
LABELS = (r"$I_0^{(1)}(z)$: no correction", r"$I^{(1)}(z)$: mirror candidate", r"$I_{corr}(z)$: zero data")

def plot_case(directory, data, processed, images, metrics, truth, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x, z = images['x'], images['z']
    vmax = max(float(images[name].max()) for name in METHODS)
    for normalized, filename in ((False, 'indicators.png'), (True, 'indicators_normalized.png')):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True, sharey=True)
        for ax, method, label in zip(axes, METHODS, LABELS):
            image = images[method]
            scale = float(image.max()) if normalized else vmax
            plotted = image / scale if scale > 0 else image * 0
            artist = ax.pcolormesh(x, z, plotted, shading='auto', vmin=0, vmax=1, cmap='viridis')
            ridge, _ = extract_ridge(image, z)
            ax.plot(x, truth, color='#ff954f', lw=2, label='True bottom')
            ax.plot(x, ridge, 'w--', lw=1, label='Column peak')
            ax.set(xlabel='x (m)', title=label)
        axes[0].set_ylabel('z (m)')
        axes[0].legend(fontsize=8)
        fig.colorbar(artist, ax=axes, label='Each panel / own maximum' if normalized else 'Energy / shared maximum')
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(x, truth, 'k-', lw=2.5, label='True bottom')
    for method, label in zip(METHODS, LABELS):
        ridge, _ = extract_ridge(images[method], z)
        ax.plot(x, ridge, lw=1, label=f'{label}, RMSE={100 * metrics[method]["rmse_m"]:.2f} cm')
    ax.set(xlabel='x (m)', ylabel='z (m)', ylim=(z[0], z[-1]))
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.savefig(directory / 'profiles.png', dpi=160)
    plt.close(fig)

    shot = len(data['source_x']) // 2
    receiver = int(np.argmin(abs(data['receiver_x'] - data['source_x'][shot])))
    time_ms = data['time'] * 1000
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True)
    axes[0].plot(time_ms, data['source_signal'], label='CQ source')
    axes[0].plot(time_ms, processed['source_signal'], label='Filtered test pulse')
    axes[0].set_ylabel('Source')
    axes[0].legend(fontsize=8)
    for ax, field, gated, label in zip(axes[1:], ('pressure', 'gradient'),
                                      ('gated_pressure', 'gated_gradient'), ('Scattered pressure', 'Scattered dp/dz')):
        values = processed[field][shot, receiver]
        ax.plot(time_ms, values, color='.55', label='Direct removed, bandpass')
        ax.plot(time_ms, processed[gated][shot, receiver], label='First-reflection window')
        peak = max(float(np.max(abs(values))), 1e-30)
        ax.plot(time_ms, processed['window'][shot, receiver] * peak, 'k:', label='Prescribed window')
        arrival = processed['reference_arrival'][shot, receiver]
        predicted_top = arrival + 2 * (config.ice_top - config.mean_bottom) / config.ice_p_speed
        ax.axvline(predicted_top * 1000, color='#ac6244', ls='--', label='Nominal first ice P return')
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel('Receiver time (ms)')
    fig.savefig(directory / 'first_reflection.png', dpi=160)
    plt.close(fig)
