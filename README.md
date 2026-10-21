# MambaHiC

MambaHiC predicts bulk Hi-C contact maps from DNA sequence and four epigenomic signals using a hybrid CNN–Mamba model. This repository includes data preprocessing, model training and evaluation, pretrained checkpoints, and analysis notebooks.

## Project Structure

```text
Mamba/
├── README.md
├── pyproject.toml
├── sample.csv
├── env/
│   ├── requirements-training-lock.txt
│   ├── setup_training_env.sh
│   └── verify_training_env.py
├── preprocessing/
│   ├── generate_4omics_pickle.py
│   ├── pairs_to_2kb_contacts.awk
│   └── pairs_to_2kb_contacts.sh
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   └── training_utils.py
├── training/
│   └── train.py
├── notebooks/
│   ├── cell_metrics/
│   │   └── cell_metrics.ipynb
│   ├── h1hesc_maps/
│   │   └── h1hesc_maps.ipynb
│   ├── hic_maps/
│   │   └── hic_maps.ipynb
│   ├── k562_maps/
│   │   └── k562_maps.ipynb
│   └── ctcf_editing/
│       └── ctcf_editing.ipynb
└── weights/
    ├── gm12878/
    │   └── best_model.pth
    └── runs_yf/
        ├── H1HESC_ex7_22/
        │   └── best_model.pth
        ├── HFF_ex7_22/
        │   └── best_model.pth
        ├── IMR90_ex7_22/
        │   └── best_model.pth
        ├── K562_ex7_22/
        │   └── best_model.pth
        └── MCF7_ex7_22/
            └── best_model.pth
```

## Data and Splits

[sample.csv](sample.csv) lists **42,005 genomic windows** with two columns: `window` and `split`. Window identifiers follow `chrN_start_end`, with coordinates in base pairs. Each window spans 1,024,000 bp, corresponding to 512 bins at 2 kb resolution.

| Split | Chromosomes | Windows |
|---|---|---:|
| Training | chr2–chr6, chr8–chr22 | 36,001 |
| Validation | chr1 | 3,557 |
| Test | chr7 | 2,447 |

The current training defaults follow this chromosome split. The training script assigns `.pkl` samples by chromosome tokens in their filenames; `sample.csv` provides the window inventory and split labels.

Each input sample combines DNA sequence with four epigenomic channels in this order: **CTCF, DNase, H3K27ac, H3K4me3**.

## Environment

| File | Purpose |
|---|---|
| [pyproject.toml](pyproject.toml) | Project metadata, core dependencies, and Python module configuration. |
| [env/requirements-training-lock.txt](env/requirements-training-lock.txt) | Pinned training runtime dependencies. |
| [env/setup_training_env.sh](env/setup_training_env.sh) | Creates the training environment and installs PyTorch and the Mamba CUDA extensions. |
| [env/verify_training_env.py](env/verify_training_env.py) | Checks CUDA kernels and runs a training step on a prepared sample. |

The setup script targets Linux with Python 3.10 and a compatible NVIDIA GPU environment. From the project root:

```bash
bash env/setup_training_env.sh
source .venv-training/bin/activate
python env/verify_training_env.py
```

## Preprocessing

| File | Purpose |
|---|---|
| [preprocessing/generate_4omics_pickle.py](preprocessing/generate_4omics_pickle.py) | Generates and validates DNA, epigenomic, and Hi-C samples in pickle format. |
| [preprocessing/pairs_to_2kb_contacts.sh](preprocessing/pairs_to_2kb_contacts.sh) | Converts Hi-C pairs into contact tables at 2 kb resolution. |
| [preprocessing/pairs_to_2kb_contacts.awk](preprocessing/pairs_to_2kb_contacts.awk) | Bins and aggregates contacts for the conversion script. |

## Model and Training

| File | Purpose |
|---|---|
| [src/data.py](src/data.py) | Pickle and HDF5 dataset loaders and sample utilities. |
| [src/metrics.py](src/metrics.py) | Correlation, error, PCC by genomic distance, and insulation-score metrics. |
| [src/models.py](src/models.py) | CNN–Mamba encoders, feature fusion, and the contact-map decoder. |
| [src/training_utils.py](src/training_utils.py) | Training loops and evaluation helpers. |
| [training/train.py](training/train.py) | Training entry point, chromosome splits, checkpoint management, and final test evaluation. |

With the environment active and prepared samples in the default data directory:

```bash
python training/train.py
```

The best checkpoint is selected by validation PCC on chr1, then evaluated on chr7 after training. Outputs are saved to `results/training/cnn_mamba_legacy/`, including `best_model.pth`, `last_checkpoint.pt`, `run_config.json`, `history.json`, and `test_metrics.json`. Use `--data-dir` and `--output-dir` to select other locations.

## Analysis Notebooks

| Notebook | Purpose |
|---|---|
| [cell_metrics](notebooks/cell_metrics/cell_metrics.ipynb) | Prediction metric distributions across cell lines. |
| [h1hesc_maps](notebooks/h1hesc_maps/h1hesc_maps.ipynb) | Observed and predicted Hi-C maps using H1hESC model weights. |
| [hic_maps](notebooks/hic_maps/hic_maps.ipynb) | GM12878 Hi-C maps and epigenomic tracks. |
| [k562_maps](notebooks/k562_maps/k562_maps.ipynb) | Observed and predicted Hi-C maps using K562 model weights. |
| [ctcf_editing](notebooks/ctcf_editing/ctcf_editing.ipynb) | TAD embeddings and CTCF signal editing analysis. |

## Pretrained Weights

Each directory contains a `best_model.pth` checkpoint.

| Cell line | Directory |
|---|---|
| GM12878 | [weights/gm12878/](weights/gm12878/) |
| H1hESC | [weights/runs_yf/H1HESC_ex7_22/](weights/runs_yf/H1HESC_ex7_22/) |
| HFF | [weights/runs_yf/HFF_ex7_22/](weights/runs_yf/HFF_ex7_22/) |
| IMR90 | [weights/runs_yf/IMR90_ex7_22/](weights/runs_yf/IMR90_ex7_22/) |
| K562 | [weights/runs_yf/K562_ex7_22/](weights/runs_yf/K562_ex7_22/) |
| MCF7 | [weights/runs_yf/MCF7_ex7_22/](weights/runs_yf/MCF7_ex7_22/) |
