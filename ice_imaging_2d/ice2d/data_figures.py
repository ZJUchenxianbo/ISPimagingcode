"""Lightweight acquisition and source/record diagnostics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_data_figures(directory, physics, records, geometry):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.linspace(-physics.horizontal_halfwidth, physics.horizontal_halfwidth, 400)
    axes[0].plot(x, geometry.evaluate(x), label='Ice bottom')
    axes[0].axhline(geometry.top, label='Ice top')
    axes[0].scatter(records['receiver_x'], np.full(len(records['receiver_x']), records['array_z']), s=8, label='Receivers')
    axes[0].scatter(records['source_x'], np.full(len(records['source_x']), records['array_z']), marker='*', label='Sources')
    axes[0].set(xlabel='x (m)', ylabel='z (m)', title='Physical acquisition')
    axes[0].legend(fontsize=7)
    axes[1].imshow(records['pressure'][len(records['source_x']) // 2], aspect='auto',
                   extent=[records['time'][0] * 1000, records['time'][-1] * 1000,
                           records['receiver_x'][-1], records['receiver_x'][0]])
    axes[1].set(xlabel='Time (ms)', ylabel='Receiver x (m)', title='Central source pressure')
    fig.savefig(directory / 'acquisition_records.png', dpi=140)
    plt.close(fig)
    dt = physics.time_step
    t = (np.arange(int(np.ceil((physics.source_center + 8 * physics.source_sigma) / dt))) + .5) * dt
    pulse = np.exp(-.5 * ((t - physics.source_center) / physics.source_sigma)**2) * np.cos(2 * np.pi * physics.source_frequency * (t - physics.source_center))
    frequency = np.fft.rfftfreq(len(t), dt)
    spectrum = abs(np.fft.rfft(pulse))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    axes[0].plot(t * 1000, pulse)
    axes[0].set(xlabel='Time (ms)', title='Source pulse')
    axes[1].plot(frequency / 1000, spectrum / max(spectrum.max(), 1e-30))
    axes[1].axvspan(physics.band_low / 1000, physics.band_high / 1000, alpha=.2)
    axes[1].set(xlim=(0, physics.band_high * 2 / 1000), xlabel='Frequency (kHz)', title='Pulse spectrum and processing band')
    fig.savefig(directory / 'source_spectrum.png', dpi=140)
    plt.close(fig)
