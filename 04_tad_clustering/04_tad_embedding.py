#!/usr/bin/env python
"""Visualize fused bin embeddings using predefined TAD boundary intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "bulk_hic_prediction").is_dir():
            return parent
    raise RuntimeError("Could not find the project root containing src/bulk_hic_prediction.")


ROOT = find_project_root()
sys.path.append(str(ROOT / "src"))
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "phase_separation" / "gm12878_legacy_best_model.pth"
DEFAULT_BOUNDARIES = (0, 75, 110, 160, 190, 360, 395, 512)
TAD_COLORS = ("#4054E8", "#F04A48", "#3FA34D", "#FFB238", "#9A3DA4", "#B45850", "#F3C3CC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run t-SNE on fused model embeddings and color bins by predefined TAD intervals."
    )
    parser.add_argument("--sample", type=Path, required=True, help="Preprocessed GM12878 .pkl sample.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--boundaries",
        default=",".join(str(value) for value in DEFAULT_BOUNDARIES),
        help="Comma-separated bin boundaries, including the first and final bin.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def parse_boundaries(value: str, bin_count: int) -> list[int]:
    boundaries = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(boundaries) < 2 or boundaries != sorted(set(boundaries)):
        raise ValueError("TAD boundaries must contain at least two unique values in ascending order.")
    if boundaries[0] != 0 or boundaries[-1] != bin_count:
        raise ValueError(f"TAD boundaries must start at 0 and end at the bin count ({bin_count}).")
    return boundaries


def load_inputs(sample_path: Path, sigma: float):
    import torch
    from scipy.ndimage import gaussian_filter

    from bulk_hic_prediction.data import load_sample

    sample = load_sample(sample_path)
    sequence = torch.from_numpy(sample["sequence_one_hot"].astype(np.float32, copy=False))
    omics = torch.from_numpy(sample["omics_signals"].astype(np.float32, copy=False))
    real_hic = np.log2(sample["hic_matrix"].astype(np.float32, copy=False) + 1.0)
    if sigma > 0:
        real_hic = gaussian_filter(real_hic, sigma=sigma)
    np.fill_diagonal(real_hic, 0)
    return sequence, omics, real_hic


def load_model(checkpoint: Path, num_omics: int, device):
    from bulk_hic_prediction.io import load_model_weights
    from bulk_hic_prediction.models import CNNMambaHybridModelLegacy

    model = CNNMambaHybridModelLegacy(
        seq_in_channels=4,
        num_omics_features=num_omics,
        d_model=128,
        fusion_out_dim=128,
        num_cnn_blocks=3,
        num_mamba_layers=3,
        decoder_bottleneck_channels=48,
        decoder_num_blocks=6,
    ).to(device)
    load_model_weights(model, checkpoint, map_location=device)
    model.eval()
    return model


def save_context_plot(output_path: Path, embedding, real_hic, predicted_hic, sample_name: str) -> None:
    import matplotlib.pyplot as plt

    positive = real_hic[real_hic > 0]
    vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    scatter = axes[0].scatter(
        embedding[:, 0], embedding[:, 1], c=np.arange(len(embedding)), cmap="viridis", s=15, alpha=0.8
    )
    axes[0].set_title("t-SNE of fused bin embeddings")
    axes[0].set_xlabel("t-SNE dimension 1")
    axes[0].set_ylabel("t-SNE dimension 2")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="Bin index")

    for ax, matrix, title in (
        (axes[1], real_hic, f"Real Hi-C\n{sample_name}"),
        (axes[2], predicted_hic, "Predicted Hi-C"),
    ):
        image = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_boundary_plot(output_path: Path, real_hic, boundaries: list[int], sample_name: str) -> None:
    import matplotlib.pyplot as plt

    positive = real_hic[real_hic > 0]
    vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(real_hic, cmap="Reds", vmin=0, vmax=vmax)
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"GM12878 Hi-C {sample_name}", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tad_plot(output_path: Path, embedding, boundaries: list[int], sample_name: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        color = TAD_COLORS[index % len(TAD_COLORS)]
        ax.scatter(
            embedding[start:end, 0],
            embedding[start:end, 1],
            s=60,
            alpha=0.7,
            color=color,
            edgecolors="black",
            linewidth=0.5,
            label=f"TAD {index + 1} ({start}-{end - 1})",
        )
    ax.set_title(f"Fused bin embedding of GM12878 {sample_name}", fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    import matplotlib
    import torch
    from sklearn.manifold import TSNE

    matplotlib.use("Agg")
    if not args.sample.is_file():
        raise FileNotFoundError(f"Sample not found: {args.sample}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    sequence, omics, real_hic = load_inputs(args.sample, args.smooth_sigma)
    boundaries = parse_boundaries(args.boundaries, int(omics.shape[-1]))
    model = load_model(args.checkpoint, int(omics.shape[0]), device)
    with torch.inference_mode():
        sequence_batch = sequence.unsqueeze(0).to(device)
        omics_batch = omics.unsqueeze(0).to(device)
        fused = model.encode_fused(sequence_batch, omics_batch)
        predicted_hic = model.decoder(fused).squeeze().detach().cpu().numpy()
    fused_bins = fused.squeeze(0).detach().cpu().numpy().T
    embedding = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(fused_bins)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = args.sample.stem
    save_context_plot(args.output_dir / "04_tad_embedding_and_hic.png", embedding, real_hic, predicted_hic, sample_name)
    save_boundary_plot(args.output_dir / "04_tad_boundaries.png", real_hic, boundaries, sample_name)
    save_tad_plot(args.output_dir / "04_tad_colored_embedding.png", embedding, boundaries, sample_name)


if __name__ == "__main__":
    main()
