from __future__ import annotations

from pathlib import Path


def load_torch_state(path: str | Path, map_location="cpu"):
    import torch

    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and any(str(key).startswith("module.") for key in state.keys()):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def load_model_weights(model, path: str | Path, *, map_location="cpu", strict: bool = True):
    state = load_torch_state(path, map_location=map_location)
    return model.load_state_dict(state, strict=strict)


def infer_num_omics_from_state(path: str | Path, model_type: str) -> int | None:
    state = load_torch_state(path, map_location="cpu")
    candidates = {
        "main": "encoder.omics_encoder.projection.weight",
        "cross_cell": "encoder.omics_encoder.projection.weight",
        "ablation_ddp": "encoder.omics_encoder.projection.weight",
        "seq2hic": "encoder.omics_encoder.projection.weight",
        "mamba_v2": "omics_encoder.downsampler.0.weight",
        "benchmark_mamba": "omics_encoder.downsampler.0.weight",
        "gm12878_legacy": "omics_encoder.downsampler.0.weight",
        "mamba_legacy": "omics_encoder.downsampler.0.weight",
        "figure04_legacy": "omics_encoder.downsampler.0.weight",
        "cnn": "encoder.omics_encoder.0.scale.0.weight",
        "reproduced_cnn": "encoder.omics_encoder.0.scale.0.weight",
        "benchmark_cnn": "encoder.omics_encoder.0.scale.0.weight",
    }
    key = candidates.get(model_type)
    if key and key in state:
        return int(state[key].shape[1])
    return None


def infer_ablation_flags_from_state(path: str | Path) -> tuple[bool, bool]:
    state = load_torch_state(path, map_location="cpu")
    has_seq = any(str(key).startswith("encoder.seq_encoder.") for key in state)
    has_omics = any(str(key).startswith("encoder.omics_encoder.") for key in state)
    ablate_dna = has_omics and not has_seq
    ablate_omics_all = has_seq and not has_omics
    return ablate_dna, ablate_omics_all


def infer_decoder_variant_from_state(path: str | Path) -> str:
    state = load_torch_state(path, map_location="cpu")
    if "decoder.projection.weight" in state:
        return "efficient"
    if "decoder.conv_start.0.weight" in state:
        return "diagonal"
    return "diagonal"


def infer_encoder_variant_from_state(path: str | Path) -> str:
    state = load_torch_state(path, map_location="cpu")
    if any("mamba_fw_blocks" in str(key) or "mamba_bw_blocks" in str(key) for key in state):
        return "blocks"
    return "single"
