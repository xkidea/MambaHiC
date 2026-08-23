#!/usr/bin/env python
"""Aggregate GPU shards and render the alpha-response and representative-map figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results"
MODE_LABELS = {
    "dual": "DNase + H3K27ac",
    "dnase": "DNase",
    "h3k27ac": "H3K27ac",
    "h3k4me3": "H3K4me3",
}
MODES = ("dual", "dnase", "h3k27ac", "h3k4me3")
IS_WINDOW_BINS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*MODES, "all", "summary"], required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def fit_power_law(alpha_values: np.ndarray, mean_values: np.ndarray):
    from scipy.optimize import curve_fit

    positive = alpha_values > 0
    x = alpha_values[positive]
    baseline = float(mean_values[alpha_values == 0][0])
    y = mean_values[positive] - baseline

    def power_law_hill(alpha, c, h):
        return c * (alpha**h)

    params, covariance = curve_fit(
        power_law_hill,
        x,
        y,
        p0=[1.0, 2.0],
        bounds=([0.0, 1.0], [np.inf, 10.0]),
    )
    return float(params[0]), float(params[1]), baseline, covariance


def aggregate_mode(mode: str, output_root: Path) -> dict:
    mode_dir = output_root / mode
    shards = sorted(mode_dir.glob("shard_*_of_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard files found in {mode_dir}")
    loaded = [np.load(path) for path in shards]
    alpha_values = loaded[0]["alpha_values"].astype(np.float64)
    sums = np.sum([item["sum_values"] for item in loaded], axis=0)
    sum_squares = np.sum([item["sum_squares"] for item in loaded], axis=0)
    counts = np.sum([item["count_values"] for item in loaded], axis=0).astype(np.int64)
    if np.any(counts == 0):
        raise RuntimeError(f"Empty alpha bin in {mode}")
    means = sums / counts
    variances = np.maximum(sum_squares / counts - means**2, 0.0)
    sem = np.sqrt(variances) / np.sqrt(counts)
    c_fit, h_fit, baseline, covariance = fit_power_law(alpha_values, means)
    fitted_values = baseline + c_fit * np.power(alpha_values, h_fit)
    residual_sum = float(np.square(means - fitted_values).sum())
    total_sum = float(np.square(means - means.mean()).sum())
    r_squared = float(1.0 - residual_sum / total_sum) if total_sum > 0 else float("nan")
    end_delta = float(means[-1] - means[0])
    response_direction = "increasing" if end_delta > 0 else "decreasing" if end_delta < 0 else "flat"
    fit_valid = response_direction == "increasing" and c_fit > 1e-8

    csv_path = mode_dir / "alpha_response.csv"
    rows = np.column_stack([alpha_values, means, sem, counts])
    np.savetxt(csv_path, rows, delimiter=",", header="alpha,mean_glcm_contrast,sem,count", comments="")

    metadata = {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "scaled_features": ["DNase", "H3K27ac"] if mode == "dual" else [MODE_LABELS[mode]],
        "sample_count": int(counts[0]),
        "alpha_count": int(len(alpha_values)),
        "alpha_min": float(alpha_values.min()),
        "alpha_max": float(alpha_values.max()),
        "fit_model": "baseline + C * alpha^h",
        "baseline_contrast": baseline,
        "C": c_fit,
        "h": h_fit,
        "r_squared": r_squared,
        "response_direction": response_direction,
        "end_minus_start": end_delta,
        "fit_valid_for_positive_response": fit_valid,
        "fit_covariance": covariance.tolist(),
        "oe_correction": True,
        "quantization": "per-prediction O/E 99.5th percentile",
        "shards": [path.name for path in shards],
    }
    (mode_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    render_response(mode, mode_dir, alpha_values, means, sem, c_fit, h_fit, fit_valid, response_direction)
    render_representatives(mode, mode_dir)
    print(json.dumps({"mode": mode, "count": int(counts[0]), "C": c_fit, "h": h_fit}))
    return metadata


def render_response(mode: str, mode_dir: Path, alpha_values, means, sem, c_fit, h_fit, fit_valid, response_direction) -> None:
    import matplotlib.pyplot as plt

    baseline = float(means[alpha_values == 0][0])
    curve_x = np.linspace(0.01, 2.0, 200)
    curve_y = c_fit * np.power(curve_x, h_fit) + baseline
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(alpha_values, means, yerr=sem, fmt="o", capsize=5, label="Mean Model Data (± SEM)")
    if fit_valid:
        fit_label = f"Fitted Power-Law (h={h_fit:.2f})"
        fit_color = "red"
        fit_style = "-"
    else:
        fit_label = f"Positive Power-Law not applicable ({response_direction} response)"
        fit_color = "0.45"
        fit_style = "--"
    ax.plot(curve_x, curve_y, fit_style, color=fit_color, label=fit_label)
    title = "Power-Law Fit to Averaged Model Response in GM12878"
    if mode != "dual":
        title += f"\n{MODE_LABELS[mode]} scaling"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Alpha (Scaling Factor)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean GLCM Contrast", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.savefig(mode_dir / "power_law_fit.png", dpi=220, bbox_inches="tight")
    fig.savefig(mode_dir / "power_law_fit.pdf", bbox_inches="tight")
    plt.close(fig)


def insulation_score(matrix: np.ndarray, window_bins: int = IS_WINDOW_BINS) -> np.ndarray:
    """Project-standard diamond/square insulation score using a bin-based window."""
    score = np.ones(matrix.shape[0], dtype=np.float32)
    for index in range(matrix.shape[0]):
        start = max(0, index - window_bins)
        stop = min(matrix.shape[0], index + window_bins + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            diamond_sum = np.sum(matrix[start:index, index + 1 : stop])
            square_sum = np.sum(matrix[start:stop, start:stop])
            value = diamond_sum / square_sum if square_sum else np.nan
            score[index] = value if np.isfinite(value) else 1.0
    return score


def parse_interval(stem: str, matrix_size: int) -> tuple[int, int, np.ndarray]:
    chromosome, start_text, end_text = stem.split("_")
    if chromosome != "chr7":
        raise ValueError(f"Unexpected representative chromosome in {stem}")
    start_bp, end_bp = int(start_text), int(end_text)
    bin_size = (end_bp - start_bp) / matrix_size
    positions_bp = start_bp + (np.arange(matrix_size, dtype=np.float64) + 0.5) * bin_size
    return start_bp, end_bp, positions_bp


def shared_limits(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    padding = max((upper - lower) * 0.05, 1e-6)
    return lower - padding, upper + padding


def render_representatives(mode: str, mode_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    path = mode_dir / "representative_maps.npz"
    if not path.exists():
        return
    with np.load(path) as data:
        alpha_values = data["alpha_values"].copy()
        maps = data["maps"].copy()
        stems = [str(value) for value in data["representative_stems"]]
    display_indices = np.linspace(0, len(alpha_values) - 1, 6, dtype=int)
    display_alphas = alpha_values[display_indices]
    alpha_norm = Normalize(vmin=float(display_alphas.min()), vmax=float(display_alphas.max()))
    alpha_cmap = plt.get_cmap("viridis")
    unperturbed_index = int(np.argmin(np.abs(alpha_values - 1.0)))

    for sample_index, stem in enumerate(stems):
        selected_maps = maps[sample_index, display_indices]
        profiles = np.stack([insulation_score(prediction) for prediction in selected_maps])
        start_bp, end_bp, positions_bp = parse_interval(stem, selected_maps.shape[-1])
        start_mb, end_mb = start_bp / 1e6, end_bp / 1e6
        genomic_ticks = np.linspace(start_mb, end_mb, 3)
        is_min, is_max = shared_limits(profiles)
        vmax = float(np.percentile(maps[sample_index, unperturbed_index], 99.5))

        profile_table = np.column_stack([positions_bp, profiles.T])
        profile_header = "position_bp," + ",".join(f"is_alpha_{alpha:.1f}" for alpha in display_alphas)
        np.savetxt(
            mode_dir / f"{stem}_is_profiles.csv",
            profile_table,
            delimiter=",",
            header=profile_header,
            comments="",
        )

        fig = plt.figure(figsize=(24, 8.5), layout="constrained")
        grid = fig.add_gridspec(
            2,
            len(display_alphas) + 1,
            width_ratios=[*([1.0] * len(display_alphas)), 0.06],
            height_ratios=[4.0, 1.35],
            hspace=0.04,
        )
        for column, (alpha, prediction, profile) in enumerate(zip(display_alphas, selected_maps, profiles)):
            map_ax = fig.add_subplot(grid[0, column])
            profile_ax = fig.add_subplot(grid[1, column], sharex=map_ax)
            image = map_ax.imshow(
                prediction,
                cmap="Reds",
                vmin=0,
                vmax=vmax,
                origin="upper",
                extent=(start_mb, end_mb, end_mb, start_mb),
            )
            map_ax.set_xlim(start_mb, end_mb)
            map_ax.set_ylim(end_mb, start_mb)
            map_ax.set_xticks(genomic_ticks)
            map_ax.tick_params(axis="x", labelbottom=False)
            map_ax.set_yticks(genomic_ticks)
            map_ax.set_title(f"α = {alpha:.1f}", fontsize=15, fontweight="bold")
            if column == 0:
                map_ax.set_ylabel("Genomic position (Mb)")
            else:
                map_ax.tick_params(axis="y", labelleft=False)

            color = alpha_cmap(alpha_norm(float(alpha)))
            profile_ax.plot(positions_bp / 1e6, profile, color=color, linewidth=1.5)
            profile_ax.set_xlim(start_mb, end_mb)
            profile_ax.set_ylim(is_min, is_max)
            profile_ax.set_xticks(genomic_ticks)
            profile_ax.set_xlabel("Position (Mb)")
            profile_ax.grid(axis="y", color="0.88", linewidth=0.7)
            if column == 0:
                profile_ax.set_ylabel("IS")
            else:
                profile_ax.tick_params(axis="y", labelleft=False)

        title = f"Predicted Hi-C Maps at Different Alpha Values in GM12878: {stem}"
        if mode != "dual":
            title += f"\n{MODE_LABELS[mode]} scaling"
        fig.suptitle(title, fontsize=18, fontweight="bold")
        colorbar_ax = fig.add_subplot(grid[0, -1])
        fig.colorbar(image, cax=colorbar_ax, label="Predicted Hi-C")
        fig.savefig(mode_dir / f"{stem}_alpha_maps.png", dpi=220, bbox_inches="tight")
        fig.savefig(mode_dir / f"{stem}_alpha_maps.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
        for alpha, profile in zip(display_alphas, profiles):
            color = alpha_cmap(alpha_norm(float(alpha)))
            ax.plot(positions_bp / 1e6, profile, color=color, linewidth=1.7, label=f"α = {alpha:.1f}")
        ax.set_xlim(start_mb, end_mb)
        ax.set_ylim(is_min, is_max)
        ax.set_xlabel("Genomic position (Mb)", fontweight="bold")
        ax.set_ylabel("Insulation score (IS)", fontweight="bold")
        ax.set_title(
            f"IS profiles across alpha: {stem}\n{MODE_LABELS[mode]} scaling, 50-bin window (100 kb)",
            fontweight="bold",
        )
        ax.grid(color="0.88", linewidth=0.8)
        ax.legend(ncol=3, frameon=False)
        fig.savefig(mode_dir / f"{stem}_is_profiles_overlay.png", dpi=220, bbox_inches="tight")
        fig.savefig(mode_dir / f"{stem}_is_profiles_overlay.pdf", bbox_inches="tight")
        plt.close(fig)


def render_all_summary(all_metadata: list[dict], output_root: Path) -> None:
    import matplotlib.pyplot as plt

    summary_path = output_root / "05_all_modes_summary.csv"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("mode,mode_label,C,h,baseline_contrast,sample_count,response_direction,fit_valid,r_squared\n")
        for item in all_metadata:
            handle.write(
                f"{item['mode']},{item['mode_label']},{item['C']:.12g},{item['h']:.12g},"
                f"{item['baseline_contrast']:.12g},{item['sample_count']},{item['response_direction']},"
                f"{item['fit_valid_for_positive_response']},{item['r_squared']:.12g}\n"
            )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [item["mode_label"] for item in all_metadata]
    values = [item["h"] if item["fit_valid_for_positive_response"] else 0.0 for item in all_metadata]
    colors = [
        color if item["fit_valid_for_positive_response"] else "#A0A0A0"
        for item, color in zip(all_metadata, ["#D55E00", "#0072B2", "#009E73", "#CC79A7"])
    ]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color="0.35", linewidth=1, linestyle="--")
    ax.set_ylabel("Hill coefficient h")
    ax.set_title("Alpha-scaling Hill coefficients in GM12878")
    ax.tick_params(axis="x", rotation=20)
    for bar, value, item in zip(bars, values, all_metadata):
        label = f"{value:.2f}" if item["fit_valid_for_positive_response"] else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_root / "05_all_modes_hill_coefficients.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_root / "05_all_modes_hill_coefficients.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        metadata = [aggregate_mode(mode, args.output_root) for mode in MODES]
        render_all_summary(metadata, args.output_root)
    elif args.mode == "summary":
        metadata = []
        for mode in MODES:
            with (args.output_root / mode / "summary.json").open(encoding="utf-8") as handle:
                metadata.append(json.load(handle))
        render_all_summary(metadata, args.output_root)
    else:
        aggregate_mode(args.mode, args.output_root)


if __name__ == "__main__":
    main()
