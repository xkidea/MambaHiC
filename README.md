# Project Structure

```text
Mamba/
|-- README.md
|-- pyproject.toml
|-- env/
|   |-- setup_training_env.sh
|   `-- verify_training_env.py
|-- preprocessing/
|   |-- generate_4omics_pickle.py
|   |-- pairs_to_2kb_contacts.awk
|   `-- pairs_to_2kb_contacts.sh
|-- src/
|   |-- data.py
|   |-- metrics.py
|   |-- models.py
|   `-- training_utils.py
|-- training/
|   `-- train.py
|-- notebooks/
|   |-- cell_metrics/
|   |   `-- cell_metrics.ipynb
|   |-- h1hesc_maps/
|   |   `-- h1hesc_maps.ipynb
|   |-- hic_maps/
|   |   `-- hic_maps.ipynb
|   |-- k562_maps/
|   |   `-- k562_maps.ipynb
|   `-- tad_embedding_ctcf_editing/
|       `-- tad_embedding_ctcf_editing.ipynb
`-- weights/
    |-- gm12878/
    |   `-- best_model.pth
    `-- runs_yf/
        |-- H1HESC_ex7_22/
        |   `-- best_model.pth
        |-- HFF_ex7_22/
        |   `-- best_model.pth
        |-- IMR90_ex7_22/
        |   `-- best_model.pth
        |-- K562_ex7_22/
        |   `-- best_model.pth
        `-- MCF7_ex7_22/
            `-- best_model.pth
```

## Root Files

| File | Contents |
|---|---|
| `README.md` | Project directory structure and file descriptions. |
| `pyproject.toml` | Python project metadata, dependencies, and module configuration. |

## Environment

| File | Contents |
|---|---|
| `env/setup_training_env.sh` | Training environment setup script. |
| `env/verify_training_env.py` | Environment verification and training-step checks. |

## Preprocessing

| File | Contents |
|---|---|
| `preprocessing/generate_4omics_pickle.py` | Generation and validation of DNA, four-omics, and Hi-C pickle samples. |
| `preprocessing/pairs_to_2kb_contacts.sh` | Shell entry point for converting Hi-C pairs to chromosome contact tables. |
| `preprocessing/pairs_to_2kb_contacts.awk` | Contact binning and aggregation used by the conversion script. |

## Source Modules

| File | Contents |
|---|---|
| `src/data.py` | Pickle and HDF5 dataset loaders and sample utilities. |
| `src/metrics.py` | Correlation, error, distance-dependent PCC, and insulation-score functions. |
| `src/models.py` | CNN-Mamba encoders, feature fusion, contact-map decoder, and model construction. |
| `src/training_utils.py` | Training loops, evaluation helpers, and JSON output utilities. |
| `training/train.py` | Model training entry point and checkpoint management. |

## Analysis Notebooks

| File | Contents |
|---|---|
| `notebooks/cell_metrics/cell_metrics.ipynb` | Cell-line prediction metric distributions. |
| `notebooks/h1hesc_maps/h1hesc_maps.ipynb` | Observed and predicted Hi-C maps using H1HESC model weights. |
| `notebooks/hic_maps/hic_maps.ipynb` | GM12878 Hi-C maps and epigenomic track visualization. |
| `notebooks/k562_maps/k562_maps.ipynb` | Observed and predicted Hi-C maps using K562 model weights. |
| `notebooks/tad_embedding_ctcf_editing/tad_embedding_ctcf_editing.ipynb` | TAD embedding visualization and CTCF signal editing analysis. |

## Model Weights

| Directory | Contents |
|---|---|
| `weights/gm12878/` | GM12878 model weights. |
| `weights/runs_yf/H1HESC_ex7_22/` | H1HESC model weights. |
| `weights/runs_yf/HFF_ex7_22/` | HFF model weights. |
| `weights/runs_yf/IMR90_ex7_22/` | IMR90 model weights. |
| `weights/runs_yf/K562_ex7_22/` | K562 model weights. |
| `weights/runs_yf/MCF7_ex7_22/` | MCF7 model weights. |
