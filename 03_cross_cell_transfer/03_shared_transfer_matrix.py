#!/usr/bin/env python
"""Evaluate one shared model on matched regions from two held-out cell lines."""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the experiment 03 transfer matrix using one model trained on "
            "HFF, GM12878, and MCF7 and evaluated on held-out H1HESC and K562."
        )
    )
    parser.add_argument("--cell-a-dir", type=Path, required=True)
    parser.add_argument("--cell-b-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Shared-model best_model.pth file.")
    parser.add_argument("--cell-a-name", default="H1HESC")
    parser.add_argument("--cell-b-name", default="K562")
    parser.add_argument("--chromosome", default="chr6")
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument("--device", default="auto", help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def common_regions(cell_a_dir: Path, cell_b_dir: Path, chromosome: str) -> list[str]:
    names_a = {path.name for path in cell_a_dir.glob(f"{chromosome}_*.pkl")}
    names_b = {path.name for path in cell_b_dir.glob(f"{chromosome}_*.pkl")}
    return sorted(names_a & names_b)


def load_inputs(path: Path, sigma: float):
    import torch
    from scipy.ndimage import gaussian_filter

    from bulk_hic_prediction.data import load_sample

    sample = load_sample(path)
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


def predict(model, sequence, omics, device) -> np.ndarray:
    import torch

    with torch.inference_mode():
        prediction = model(sequence.unsqueeze(0).to(device), omics.unsqueeze(0).to(device))
    return prediction.squeeze(0).detach().cpu().numpy()


def correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    from scipy.stats import pearsonr

    pred_flat = np.nan_to_num(prediction.reshape(-1).astype(np.float64, copy=False))
    target_flat = np.nan_to_num(target.reshape(-1).astype(np.float64, copy=False))
    if pred_flat.size < 2 or np.std(pred_flat) == 0 or np.std(target_flat) == 0:
        return float("nan")
    return float(pearsonr(pred_flat, target_flat)[0])


def save_heatmap(path: Path, matrix: np.ndarray, cell_a: str, cell_b: str, chromosome: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        linewidths=0.5,
        xticklabels=[f"Real {cell_a}", f"Real {cell_b}"],
        yticklabels=[f"Predicted {cell_a}", f"Predicted {cell_b}"],
        ax=ax,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(f"Cross-cell-line mean PCC matrix on {chromosome}")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    import matplotlib
    import torch

    matplotlib.use("Agg")
    for path in (args.cell_a_dir, args.cell_b_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Cell-line data directory not found: {path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    regions = common_regions(args.cell_a_dir, args.cell_b_dir, args.chromosome)
    if args.max_regions is not None:
        regions = regions[: args.max_regions]
    if not regions:
        raise RuntimeError(f"No matched {args.chromosome} regions were found.")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    first_sequence, first_omics, _ = load_inputs(args.cell_a_dir / regions[0], args.smooth_sigma)
    model = load_model(args.checkpoint, int(first_omics.shape[0]), device)

    pcc_a_a: list[float] = []
    pcc_a_b: list[float] = []
    pcc_b_a: list[float] = []
    pcc_b_b: list[float] = []
    for region in regions:
        sequence_a, omics_a, real_a = load_inputs(args.cell_a_dir / region, args.smooth_sigma)
        sequence_b, omics_b, real_b = load_inputs(args.cell_b_dir / region, args.smooth_sigma)
        predicted_a = predict(model, sequence_a, omics_a, device)
        predicted_b = predict(model, sequence_b, omics_b, device)
        pcc_a_a.append(correlation(predicted_a, real_a))
        pcc_a_b.append(correlation(predicted_a, real_b))
        pcc_b_a.append(correlation(predicted_b, real_a))
        pcc_b_b.append(correlation(predicted_b, real_b))

    matrix = np.asarray(
        [
            [np.nanmean(pcc_a_a), np.nanmean(pcc_a_b)],
            [np.nanmean(pcc_b_a), np.nanmean(pcc_b_b)],
        ],
        dtype=float,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"03_transfer_pcc_matrix_{args.chromosome}.png"
    save_heatmap(image_path, matrix, args.cell_a_name, args.cell_b_name, args.chromosome)
    payload = {
        "training_cell_lines": ["HFF", "GM12878", "MCF7"],
        "held_out_cell_lines": [args.cell_a_name, args.cell_b_name],
        "chromosome": args.chromosome,
        "matched_regions": len(regions),
        "rows": [f"Predicted {args.cell_a_name}", f"Predicted {args.cell_b_name}"],
        "columns": [f"Real {args.cell_a_name}", f"Real {args.cell_b_name}"],
        "mean_pcc": matrix.tolist(),
    }
    summary_path = args.output_dir / f"03_transfer_pcc_matrix_{args.chromosome}.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
