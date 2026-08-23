# Phase-Separation Reproduction

This experiment measures how predicted GM12878 Hi-C texture changes when selected
epigenomic signals are multiplied by alpha values from 0.0 to 2.0. The analysis uses
1,224 chromosome 7 windows and fits `baseline + C * alpha^h` to mean O/E GLCM contrast.

Only four retained modes are included:

- `dnase`: DNase scaling
- `h3k4me3`: H3K4me3 scaling
- `h3k27ac`: H3K27ac scaling
- `dual`: joint DNase and H3K27ac scaling

`results/` contains the final figures, response tables, and fit summaries. Intermediate
GPU shards, representative-map arrays, input data, caches, and logs are intentionally
excluded.

To rerun all four modes on six GPUs:

```bash
export DATA_DIR=/path/to/gm12878/preprocessed/pkl
bash 05_phase_separation_reproduction/run_all.sh
```

Set `PYTHON` to select a Python executable or `MODES` to run a subset. The default model
weights are read from `checkpoints/phase_separation/gm12878_legacy_best_model.pth`.
