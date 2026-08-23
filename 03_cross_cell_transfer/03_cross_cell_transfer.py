#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "bulk_hic_prediction").is_dir():
            return parent
    raise RuntimeError("Could not find organized project root containing src/bulk_hic_prediction.")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-cell transfer and recreate the experiment 03 panels.")
    parser.add_argument("--base-data-dir", required=True, help="Directory containing one subdirectory per cell line.")
    parser.add_argument(
        "--weights-dir",
        required=True,
        help="Directory containing <cell_line>/best_model.pth. Cross-cell checkpoints are not bundled.",
    )
    parser.add_argument("--cell-lines", default="HFF_hg38,H1HESC_hg38,K562_hg38,MCF7_hg38,GM12878_hg38")
    parser.add_argument("--chromosome", default="chr7")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--omics-indices", default=None)
    parser.add_argument("--make-cross-matrix", action="store_true", help="Evaluate every checkpoint against every cell line.")
    parser.add_argument("--example-count", type=int, default=1)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    return parser.parse_args()


def psnr(pred, target) -> float:
    import numpy as np

    mse = float(np.mean((pred - target) ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(np.nanmax(target) - np.nanmin(target))
    if data_range <= 0:
        data_range = 1.0
    return float(20 * np.log10(data_range / np.sqrt(mse)))


def add_psnr(rows: list[dict[str, float]], preds, targets) -> list[dict[str, float]]:
    for row, pred, target in zip(rows, preds, targets):
        row["psnr"] = psnr(pred, target)
    return rows


def display_label(cell_line: str) -> str:
    return cell_line.removesuffix("_hg38")


def infer_and_build_model(weight_path: Path, target_len: int, selected_omics: int, device):
    from bulk_hic_prediction.io import (
        infer_ablation_flags_from_state,
        infer_decoder_variant_from_state,
        infer_encoder_variant_from_state,
        infer_num_omics_from_state,
        load_model_weights,
    )
    from bulk_hic_prediction.models import build_model

    state_omics = infer_num_omics_from_state(weight_path, "cross_cell")
    if state_omics is not None and state_omics != selected_omics:
        raise ValueError(
            f"{weight_path} expects {state_omics} omics tracks but selected data has {selected_omics}. "
            "Pass --omics-indices to select matching tracks."
        )
    ablate_dna, ablate_omics_all = infer_ablation_flags_from_state(weight_path)
    decoder_variant = infer_decoder_variant_from_state(weight_path)
    encoder_variant = infer_encoder_variant_from_state(weight_path)
    model = build_model(
        "cross_cell",
        state_omics or selected_omics,
        target_len=target_len,
        ablate_dna=ablate_dna,
        ablate_omics_all=ablate_omics_all,
        decoder_variant=decoder_variant,
        encoder_variant=encoder_variant,
    ).to(device)
    load_model_weights(model, weight_path, map_location=device)
    return model


def plot_metric_boxplots(output_path: Path, rows_by_cell: dict[str, list[dict[str, float]]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [("pcc", "PCC"), ("spearman", "Spearman"), ("psnr", "PSNR")]
    labels = list(rows_by_cell)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, (metric_key, title) in zip(axes, metrics):
        values = [[row[metric_key] for row in rows_by_cell[label] if np.isfinite(row[metric_key])] for label in labels]
        bp = ax.boxplot(values, patch_artist=True, showfliers=False, tick_labels=[display_label(label) for label in labels])
        for patch in bp["boxes"]:
            patch.set_facecolor("#72B7B2")
            patch.set_alpha(0.75)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=25)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_examples(output_path: Path, examples: dict[str, tuple[object, object]], example_count: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(examples)
    if not labels:
        return
    rows = sum(min(example_count, len(examples[label][0])) for label in labels)
    if rows <= 0:
        return
    fig, axes = plt.subplots(rows, 2, figsize=(6.5, 3.1 * rows))
    axes = np.asarray(axes).reshape(rows, 2)
    row_idx = 0
    for label in labels:
        preds, targets = examples[label]
        for idx in range(min(example_count, len(preds))):
            target = targets[idx]
            pred = preds[idx]
            positive = target[target > 0]
            vmax = np.percentile(positive, 99) if positive.size else max(float(np.nanmax(target)), 1.0)
            label_text = display_label(label)
            for ax, title, matrix in [(axes[row_idx, 0], f"{label_text} target", target), (axes[row_idx, 1], f"{label_text} prediction", pred)]:
                ax.imshow(matrix, cmap="Reds", vmin=0, vmax=vmax)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
            row_idx += 1
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_matrix(output_path: Path, matrix: list[list[float]], row_labels: list[str], col_labels: list[str]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.3 * len(col_labels) + 3, 1.1 * len(row_labels) + 2.5))
    im = ax.imshow(arr, cmap="viridis", vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    ax.set_xticks(np.arange(len(col_labels)), labels=[display_label(label) for label in col_labels], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), labels=[display_label(label) for label in row_labels])
    ax.set_xlabel("Evaluation data")
    ax.set_ylabel("Checkpoint")
    ax.set_title("Cross-cell mean PCC matrix", fontweight="bold")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", color="white" if arr[i, j] < np.nanmean(arr) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from bulk_hic_prediction.data import HiCDataAndOmicsDataset, list_pkl_files, load_sample, parse_indices
    from bulk_hic_prediction.training_utils import evaluate_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_lines = [part.strip() for part in args.cell_lines.split(",") if part.strip()]
    selected_indices = parse_indices(args.omics_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows_by_cell = {}
    examples = {}
    summaries = {}
    dataset_cache = {}
    loader_cache = {}
    target_len_cache = {}
    selected_omics_cache = {}

    for cell_line in cell_lines:
        data_dir = Path(args.base_data_dir) / cell_line
        files = list_pkl_files(data_dir, args.chromosome, args.max_samples)
        weight_path = Path(args.weights_dir) / cell_line / "best_model.pth"
        if not files or not weight_path.exists():
            print(f"skip {cell_line}: files={len(files)} weight_exists={weight_path.exists()}")
            continue
        sample = load_sample(files[0])
        target_len_cache[cell_line] = int(sample["omics_signals"].shape[-1])
        sample_omics = int(sample["omics_signals"].shape[0])
        selected_omics_cache[cell_line] = len(selected_indices) if selected_indices is not None else sample_omics
        dataset = HiCDataAndOmicsDataset(files, zero_diagonal=True, omics_indices=selected_indices)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        dataset_cache[cell_line] = dataset
        loader_cache[cell_line] = loader

        model = infer_and_build_model(weight_path, target_len_cache[cell_line], selected_omics_cache[cell_line], device)
        summary, rows, preds, targets = evaluate_model(model, loader, device)
        rows_by_cell[cell_line] = add_psnr(rows, preds, targets)
        summaries[cell_line] = summary
        examples[cell_line] = (preds, targets)

    if not rows_by_cell:
        raise RuntimeError("No matched cross-cell evaluations completed. Check --base-data-dir and --weights-dir.")

    plot_metric_boxplots(output_dir / f"03_cross_cell_{args.chromosome}_metric_boxplots.png", rows_by_cell)
    plot_examples(output_dir / f"03_cross_cell_{args.chromosome}_examples.png", examples, args.example_count)

    cross_matrix_payload = None
    if args.make_cross_matrix:
        row_labels = []
        col_labels = list(dataset_cache)
        matrix = []
        for weight_cell in col_labels:
            weight_path = Path(args.weights_dir) / weight_cell / "best_model.pth"
            row = []
            row_labels.append(weight_cell)
            for data_cell in col_labels:
                model = infer_and_build_model(weight_path, target_len_cache[data_cell], selected_omics_cache[data_cell], device)
                _, rows, _, _ = evaluate_model(model, loader_cache[data_cell], device)
                values = [r["pcc"] for r in rows]
                row.append(float(sum(values) / len(values)) if values else float("nan"))
            matrix.append(row)
        plot_cross_matrix(output_dir / f"03_cross_cell_{args.chromosome}_pcc_matrix.png", matrix, row_labels, col_labels)
        cross_matrix_payload = {"rows": row_labels, "columns": col_labels, "mean_pcc": matrix}

    with open(output_dir / f"03_cross_cell_{args.chromosome}_sample_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell_line", "sample_index", "pcc", "spearman", "mae", "mse", "psnr"])
        writer.writeheader()
        for cell_line, rows in rows_by_cell.items():
            for idx, row in enumerate(rows):
                writer.writerow({"cell_line": cell_line, "sample_index": idx, **row})
    with open(output_dir / f"03_cross_cell_{args.chromosome}_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"summaries": summaries, "cross_matrix": cross_matrix_payload}, handle, indent=2)


if __name__ == "__main__":
    main()
