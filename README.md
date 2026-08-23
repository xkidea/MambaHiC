# COCO-Mamba Hi-C Prediction

This repository contains the COCO-Mamba model utilities and six retained
experiments for predicting Hi-C contact maps from DNA sequence and epigenomic
signals.

## Project Structure

```text
.
|-- 01_real_window_prediction/        Prediction on a representative chr7 window
|-- 02_corigami_comparison/           Comparison with a reproduced C.Origami model
|-- 03_cross_cell_transfer/           Cross-cell-line evaluation and transfer results
|-- 04_tad_clustering/                TAD-colored fused-embedding visualization
|-- 05_phase_separation_reproduction/ Epigenomic signal-scaling analysis
|-- 06_interpretability_and_editing/  Global and local attribution analyses
|-- checkpoints/
|   |-- benchmark/                    COCO-Mamba and C.Origami benchmark checkpoints
|   |-- main/                         Final COCO-Mamba checkpoint
|   `-- phase_separation/             GM12878 legacy checkpoint used by experiments 04-05
|-- data/                             Local example input
|-- src/
|   |-- bulk_hic_prediction/          Data, model, metric, plotting, and training utilities
|   `-- mamba_ssm/                    Local pure-Python Mamba fallback
|-- requirement.txt                   Python dependencies
`-- selected codes.zip                Pre-existing archive (left unchanged)
```

Each experiment directory contains its analysis code and saved figures. The
cross-cell directory includes the retained five-cell-line metrics, H1HESC/K562
transfer examples, and the chromosome 6 transfer matrix. Its raw data and
cross-cell checkpoints are intentionally not bundled. The TAD visualization
uses predefined genomic boundary intervals to color a t-SNE embedding; it is
not an unsupervised clustering result. Experiment 05 retains only DNase,
H3K4me3, H3K27ac, and dual (DNase + H3K27ac) scaling.

## Setup

```bash
pip install -r requirement.txt
```

CUDA is recommended for model evaluation. Experiments 03 and 04 accept external
preprocessed inputs through their command-line arguments; run each script with
`--help` for details. Experiment 03 provides separate scripts for the
five-cell-line metrics and the shared-model H1HESC/K562 transfer matrix.
