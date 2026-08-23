#!/usr/bin/env python
"""Run one GPU shard of the GM12878 alpha perturbation experiment."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "checkpoints" / "phase_separation" / "gm12878_legacy_best_model.pth"
FEATURE_NAMES = ("CTCF", "DNase", "H3K27ac", "H3K4me3")
MODE_CHANNELS = {
    "dual": (1, 2),
    "dnase": (1,),
    "h3k27ac": (2,),
    "h3k4me3": (3,),
}
MODE_LABELS = {
    "dual": "DNase + H3K27ac",
    "dnase": "DNase",
    "h3k27ac": "H3K27ac",
    "h3k4me3": "H3K4me3",
}
ALPHAS = np.arange(0.0, 2.0 + 0.1 / 2.0, 0.1, dtype=np.float64)
SAMPLE_RE = re.compile(r"^chr7_(\d+)_(\d+)\.pkl$")
REPRESENTATIVE_STEMS = (
    "chr7_104576000_105600000",
    "chr7_21248000_22272000",
    "chr7_15616000_16640000",
    "chr7_39168000_40192000",
    "chr7_80000000_81024000",
    "chr7_119040000_120064000",
    "chr7_142592000_143616000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_CHANNELS), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--alpha-count", type=int, default=None)
    parser.add_argument("--skip-representatives", action="store_true")
    parser.add_argument("--representatives-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def selected_files(data_dir: Path, max_samples: int | None = None) -> list[Path]:
    files: list[tuple[int, Path]] = []
    for path in data_dir.glob("chr7_*.pkl"):
        match = SAMPLE_RE.match(path.name)
        if match is None:
            continue
        start = int(match.group(1))
        # The source notebook used the 64-bin-stride subset: starts are 128 kb apart.
        if start % 128_000 == 0:
            files.append((start, path))
    files.sort(key=lambda item: item[0])
    selected = [path for _, path in files]
    if len(selected) != 1224:
        raise RuntimeError(f"Expected 1224 original-stride chr7 samples, found {len(selected)}")
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def load_input(path: Path):
    import torch

    with path.open("rb") as handle:
        sample = pickle.load(handle)
    seq = torch.from_numpy(sample["sequence_one_hot"].astype(np.float32, copy=False))
    omics = torch.from_numpy(sample["omics_signals"].astype(np.float32, copy=False))
    if tuple(omics.shape) != (4, 512):
        raise RuntimeError(f"Unexpected omics shape {tuple(omics.shape)} in {path}")
    return seq, omics


def calculate_expected_hic(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    expected_values = np.zeros(n)
    for distance in range(n):
        diagonal = np.diagonal(matrix, offset=distance)
        if len(diagonal) > 0:
            expected_values[distance] = np.nanmean(diagonal)
    return expected_values


def calculate_oe_matrix(matrix: np.ndarray) -> np.ndarray:
    expected_values = calculate_expected_hic(matrix)
    rows, cols = np.indices(matrix.shape)
    distances = np.abs(rows - cols)
    expected_matrix = expected_values[distances]
    return np.divide(matrix, expected_matrix, where=expected_matrix != 0)


def calculate_glcm_contrast(matrix: np.ndarray) -> float:
    from skimage.feature import graycomatrix, graycoprops

    oe_matrix = calculate_oe_matrix(matrix)
    finite = oe_matrix[np.isfinite(oe_matrix)]
    if finite.size == 0:
        return float("nan")
    oe_matrix = np.nan_to_num(oe_matrix, posinf=np.nanmax(finite))
    vmax = np.percentile(oe_matrix, 99.5)
    clipped_matrix = np.clip(oe_matrix, 0, vmax)
    if vmax > 0:
        scaled_matrix = (clipped_matrix / vmax * 255).astype(np.uint8)
    else:
        scaled_matrix = np.zeros_like(clipped_matrix, dtype=np.uint8)
    glcm = graycomatrix(
        scaled_matrix,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    return float(graycoprops(glcm, "contrast").mean())


def build_model(device, weights: Path):
    import torch

    # Appending keeps an installed CUDA mamba_ssm ahead of the local CPU fallback.
    sys.path.append(str(PROJECT_ROOT / "src"))
    from bulk_hic_prediction.models import CNNMambaHybridModelLegacy

    model = CNNMambaHybridModelLegacy(
        seq_in_channels=4,
        num_omics_features=4,
        d_model=128,
        fusion_out_dim=128,
        num_cnn_blocks=3,
        num_mamba_layers=3,
        decoder_bottleneck_channels=48,
        decoder_num_blocks=6,
    )
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def predict_one(model, seq, omics, alpha_values, channels, device):
    import torch

    seq_batch = seq.unsqueeze(0).to(device)
    omics_batch = omics.unsqueeze(0).to(device)
    predictions = []
    with torch.inference_mode():
        for alpha in alpha_values:
            modified = omics_batch.clone()
            modified[:, list(channels), :] *= float(alpha)
            predictions.append(model(seq_batch, modified)[0].detach().cpu().numpy())
    return np.stack(predictions, axis=0)


def save_representatives(
    model,
    files,
    mode: str,
    alpha_values,
    channels,
    device,
    output_root: Path,
    data_dir: Path,
    weights: Path,
) -> None:
    reps = []
    for stem in REPRESENTATIVE_STEMS:
        path = next((candidate for candidate in files if candidate.stem == stem), None)
        if path is None:
            raise RuntimeError(f"Representative sample not found: {stem}")
        seq, omics = load_input(path)
        reps.append(predict_one(model, seq, omics, alpha_values, channels, device))

    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    pending_path = mode_dir / "representative_maps.pending.npz"
    np.savez_compressed(
        pending_path,
        alpha_values=alpha_values,
        maps=np.stack(reps, axis=0),
        representative_stems=np.asarray(REPRESENTATIVE_STEMS),
    )
    pending_path.replace(mode_dir / "representative_maps.npz")

    metadata = {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "scaled_channels": list(channels),
        "scaled_features": [FEATURE_NAMES[index] for index in channels],
        "representative_stems": list(REPRESENTATIVE_STEMS),
        "representative_count": len(REPRESENTATIVE_STEMS),
        "alpha_values": alpha_values.tolist(),
        "data_dir": str(data_dir),
        "weights": str(weights),
    }
    (mode_dir / "representative_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_id < args.shard_count:
        raise ValueError("Invalid shard id/count")
    if args.alpha_count is None:
        alpha_values = ALPHAS
    else:
        if args.alpha_count < 1 or args.alpha_count > len(ALPHAS):
            raise ValueError("alpha-count must be between 1 and 21")
        alpha_values = ALPHAS[: args.alpha_count]

    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to write CPU reproduction outputs")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    model = build_model(device, args.weights)
    files = selected_files(args.data_dir, args.max_samples)
    shard_files = files[args.shard_id :: args.shard_count]
    channels = MODE_CHANNELS[args.mode]

    if args.representatives_only:
        save_representatives(
            model,
            files,
            args.mode,
            alpha_values,
            channels,
            device,
            args.output_root,
            args.data_dir,
            args.weights,
        )
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "representatives": len(REPRESENTATIVE_STEMS),
                    "alpha_count": len(alpha_values),
                }
            )
        )
        return

    class InputDataset(Dataset):
        def __len__(self):
            return len(shard_files)

        def __getitem__(self, index):
            seq, omics = load_input(shard_files[index])
            return seq, omics

    loader = DataLoader(InputDataset(), batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    sum_values = np.zeros(len(alpha_values), dtype=np.float64)
    sum_squares = np.zeros(len(alpha_values), dtype=np.float64)
    count_values = np.zeros(len(alpha_values), dtype=np.int64)

    for seq_batch, omics_batch in loader:
        seq_batch = seq_batch.to(device, non_blocking=True)
        omics_batch = omics_batch.to(device, non_blocking=True)
        with torch.inference_mode():
            for alpha_index, alpha in enumerate(alpha_values):
                modified = omics_batch.clone()
                modified[:, list(channels), :] *= float(alpha)
                predictions = model(seq_batch, modified).detach().cpu().numpy()
                values = np.asarray(
                    [calculate_glcm_contrast(pred) for pred in predictions],
                    dtype=np.float64,
                )
                finite = np.isfinite(values)
                sum_values[alpha_index] += float(values[finite].sum())
                sum_squares[alpha_index] += float(np.square(values[finite]).sum())
                count_values[alpha_index] += int(finite.sum())

    mode_dir = args.output_root / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    shard_path = mode_dir / f"shard_{args.shard_id:02d}_of_{args.shard_count:02d}.npz"
    np.savez_compressed(
        shard_path,
        alpha_values=alpha_values,
        sum_values=sum_values,
        sum_squares=sum_squares,
        count_values=count_values,
        shard_id=args.shard_id,
        shard_count=args.shard_count,
    )

    if args.shard_id == 0 and not args.skip_representatives:
        save_representatives(
            model,
            files,
            args.mode,
            alpha_values,
            channels,
            device,
            args.output_root,
            args.data_dir,
            args.weights,
        )

    metadata = {
        "mode": args.mode,
        "mode_label": MODE_LABELS[args.mode],
        "scaled_channels": list(channels),
        "scaled_features": [FEATURE_NAMES[index] for index in channels],
        "data_dir": str(args.data_dir),
        "weights": str(args.weights),
        "sample_count_total": len(files),
        "sample_count_shard": len(shard_files),
        "stride_filter": "start_bp % 128000 == 0",
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
        "gpu_id": args.gpu_id,
        "alpha_values": alpha_values.tolist(),
        "oe_correction": True,
        "quantization": "per-prediction O/E 99.5th percentile",
    }
    (mode_dir / f"shard_{args.shard_id:02d}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"shard": args.shard_id, "mode": args.mode, "samples": len(shard_files)}))


if __name__ == "__main__":
    main()
