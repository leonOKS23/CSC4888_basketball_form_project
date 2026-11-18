"""Reproduce the diagnostic for the known late-contact release failure."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inspect_data import ROOT, release_frame


if __name__ == '__main__':
    trial = json.loads((ROOT / 'data/raw/2025-12-18/P0001/BB_FT_P0001_T0036.json').read_text())
    frames = trial['tracking']
    ball = np.asarray([f['data']['ball'] for f in frames], float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for side in ('LEFT', 'RIGHT'):
        wrist = np.asarray([f['data']['player'][side + '_WRIST'] for f in frames], float)
        axes[0].plot(np.linalg.norm(ball - wrist, axis=1), label=side + ' wrist distance')
        axes[1].plot(wrist[:, 2], label=side + ' wrist z')
    axes[0].axhline(.79, color='gray', ls=':')
    axes[1].plot(ball[:, 2], label='ball z')
    for ax in axes:
        ax.axvline(release_frame(trial), color='red', ls='--', label='release estimate')
        ax.legend()
        ax.set_ylabel('Feet')
    axes[1].set_xlabel('Frame index')
    fig.suptitle('P0001 T0036 — release estimate at final frame; review required')
    fig.tight_layout()
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / 'data/processed/release_outlier.png', dpi=140)
