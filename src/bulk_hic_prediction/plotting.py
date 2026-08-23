from __future__ import annotations

from pathlib import Path

import numpy as np


def save_heatmap_triplet(path: str | Path, real: np.ndarray, pred: np.ndarray, diff: np.ndarray | None = None, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if diff is None:
        diff = pred - real
    vmax = np.percentile(real[real > 0], 99) if np.any(real > 0) else max(float(np.max(real)), 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    if title:
        fig.suptitle(title)
    for ax, matrix, name, cmap in [
        (axes[0], real, "target", "Reds"),
        (axes[1], pred, "prediction", "Reds"),
        (axes[2], diff, "prediction - target", "coolwarm"),
    ]:
        if name == "prediction - target":
            lim = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1e-6)
            im = ax.imshow(matrix, cmap=cmap, vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_distance_plot(path: str | Path, series: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values in series.items():
        x = np.arange(1, len(values) + 1)
        ax.plot(x, values, label=label, linewidth=2)
    ax.set_xlabel("Genomic distance (bins)")
    ax.set_ylabel("Pearson correlation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
