#!/usr/bin/env python
"""Persist the local-attribution values missing from notebook 06.
1. omics attribution (GPU): 50-step IG -> 06_local_attribution_omics.npy + redraw
2. seq attribution (CPU, 2 processes x 16 threads): 50 IG steps split into two 25-step blocks, then merged
   -> 06_local_attribution_seq.npy + redraw + stats JSON
Checkpoint: an existing npy skips its step.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
PKG = BASE / '..'
sys.path.insert(0, str(PKG / 'src'))
from mamba_ssm import Mamba

nb01 = json.load(open(BASE / '..' / '01_real_window_prediction' / '01_real_window_prediction.ipynb'))
exec(''.join(nb01['cells'][2]['source']), globals())

WEIGHTS = PKG / 'checkpoints' / 'main' / 'best_model.pth'
DATA_DIR = Path('/mnt/nfs/clgou/Mamba/datasets/preprocessed_data_omic_512')
OUT = BASE
N_STEPS = 50

OMICS_NAMES = ['CTCF', 'DNase', 'H3K27ac', 'H3K4me3']


def load_window(f):
    with open(f, 'rb') as fh:
        d = pickle.load(fh)
    seq = torch.from_numpy(d['sequence_one_hot'].astype(np.float32))
    omics = torch.from_numpy(d['omics_signals'].astype(np.float32))
    return seq, omics


def main():
    meta = json.load(open(OUT / '06_local_target.json'))
    ti, tj = meta['target_i'], meta['target_j']
    window_file = str(DATA_DIR / f"{meta['window']}.pkl")
    seq, omics = load_window(window_file)
    print(f'window {meta["window"]}, target bin ({ti},{tj})', flush=True)

    model = MultiModalSeq2HiCModel(num_omics_features=4)
    st = torch.load(str(WEIGHTS), map_location='cpu')
    if all(k.startswith('module.') for k in st.keys()):
        st = {k.replace('module.', ''): v for k, v in st.items()}
    model.load_state_dict(st)

    # ---------- 1. omics attribution (GPU) ----------
    if not (OUT / '06_local_attribution_omics.npy').exists():
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        model_gpu = model.cuda().eval()
        seq_b = seq.unsqueeze(0).cuda()
        omics_b = omics.unsqueeze(0).cuda()

        def fwd_omics(omics_in, seq_in):
            return model_gpu(seq_in, omics_in)[:, ti, tj]

        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(fwd_omics)
        t0 = time.time()
        attr = ig.attribute(omics_b, baselines=torch.zeros_like(omics_b),
                            additional_forward_args=(seq_b,), n_steps=N_STEPS,
                            internal_batch_size=1)
        omics_attr_np = attr.squeeze(0).abs().cpu().numpy()
        np.save(OUT / '06_local_attribution_omics.npy', omics_attr_np)
        print(f'omics attribution done ({time.time()-t0:.0f}s), npy saved', flush=True)
        del model_gpu, seq_b, omics_b
        torch.cuda.empty_cache()
    else:
        print('omics npy exists, skipping', flush=True)
        omics_attr_np = np.load(OUT / '06_local_attribution_omics.npy')

    # ---------- 2. seq attribution (CPU, 2 processes x 16 threads) ----------
    if not (OUT / '06_local_attribution_seq.npy').exists():
        print('seq attribution: 2 processes (25 steps x 16 threads each), ~40 min expected', flush=True)
        seq_np = seq.numpy()       # (4, 1024000)
        omics_np = omics.numpy()   # (4, 512)
        half = N_STEPS // 2
        procs = []
        for proc_id, (lo, hi) in enumerate([(0, half), (half, N_STEPS)]):
            cmd = [sys.executable, str(BASE / 'seq_attr_worker.py'),
                   '--proc', str(proc_id), '--lo', str(lo), '--hi', str(hi),
                   '--ti', str(ti), '--tj', str(tj),
                   '--seq', str(OUT / 'seq_input.npy'), '--omics', str(OUT / 'omics_input.npy'),
                   '--out', str(OUT / f'seq_grad_part_{proc_id}.npy')]
            procs.append(subprocess.Popen(cmd))
        # save inputs for workers first
        np.save(OUT / 'seq_input.npy', seq_np)
        np.save(OUT / 'omics_input.npy', omics_np)
        for p in procs:
            p.wait()
        # merge the two gradient parts
        part0 = np.load(OUT / 'seq_grad_part_0.npy')
        part1 = np.load(OUT / 'seq_grad_part_1.npy')
        grad_sum = part0 + part1
        seq_attr = seq_np * grad_sum / N_STEPS          # IG: (x - x0) * mean(grad)
        seq_attr_np = np.abs(seq_attr).sum(axis=0)      # (1024000,) aggregated over bases
        np.save(OUT / '06_local_attribution_seq.npy', seq_attr_np)
        print('seq attribution done, npy saved', flush=True)
        os.remove(OUT / 'seq_grad_part_0.npy')
        os.remove(OUT / 'seq_grad_part_1.npy')
        os.remove(OUT / 'seq_input.npy')
        os.remove(OUT / 'omics_input.npy')
    else:
        print('seq npy exists, skipping', flush=True)
        seq_attr_np = np.load(OUT / '06_local_attribution_seq.npy')

    # ---------- 3. redraw figures (from npy, consistent with saved results) ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # omics heatmap
    attr_pos = omics_attr_np[omics_attr_np > 0]
    vmax_attr = float(np.percentile(attr_pos, 99)) if attr_pos.size else 1.0
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(omics_attr_np, cmap='viridis', aspect='auto', vmin=0, vmax=vmax_attr)
    ax.set_yticks(range(4)); ax.set_yticklabels(OMICS_NAMES)
    ax.set_xlabel('Genomic bin (2kb)')
    ax.set_title(f'Local attribution of Hi-C bin ({ti},{tj}) — omics inputs', fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(OUT / '06_local_attribution_omics.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    # seq curve
    bin_size = seq_attr_np.shape[0] // 512
    seq_binned = seq_attr_np[:512*bin_size].reshape(512, bin_size).mean(axis=1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(seq_binned, color='#2c3e50', lw=1)
    ax.set_xlabel('Genomic bin (2kb)'); ax.set_ylabel('|attr|')
    ax.set_title(f'Local attribution of Hi-C bin ({ti},{tj}) — DNA sequence (2kb-binned)', fontweight='bold')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / '06_local_attribution_seq.png', dpi=220, bbox_inches='tight'); plt.close(fig)

    # stats JSON
    stats = {'window': meta['window'], 'target_bin': [ti, tj],
             'omics_attr_shape': list(omics_attr_np.shape),
             'omics_attr_sum': float(np.abs(omics_attr_np).sum()),
             'seq_attr_sum': float(seq_attr_np.sum()),
             'omics_by_track': {n: float(np.abs(omics_attr_np[i]).sum())
                                for i, n in enumerate(OMICS_NAMES)}}
    with open(OUT / '06_local_attribution_stats.json', 'w') as fh:
        json.dump(stats, fh, indent=2)
    print('stats JSON saved', flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
