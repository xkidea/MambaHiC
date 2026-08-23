#!/usr/bin/env python
"""Seq-attribution worker: computes the sum of IG gradients over step range [lo, hi) (CPU).
Usage: python seq_attr_worker.py --proc 0 --lo 0 --hi 25 --ti 438 --tj 496 \
      --seq <seq.npy> --omics <omics.npy> --out <part.npy>
The main process sums the two gradient parts and divides by N_STEPS to obtain the full IG (by additivity of the integral).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(16)

BASE = Path(__file__).resolve().parent
PKG = BASE / '..'
sys.path.insert(0, str(PKG / 'src'))
from mamba_ssm import Mamba

nb01 = json.load(open(BASE / '..' / '01_real_window_prediction' / '01_real_window_prediction.ipynb'))
exec(''.join(nb01['cells'][2]['source']), globals())

WEIGHTS = PKG / 'checkpoints' / 'main' / 'best_model.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc', type=int, required=True)
    ap.add_argument('--lo', type=int, required=True)
    ap.add_argument('--hi', type=int, required=True)
    ap.add_argument('--ti', type=int, required=True)
    ap.add_argument('--tj', type=int, required=True)
    ap.add_argument('--seq', required=True)
    ap.add_argument('--omics', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    seq_np = np.load(args.seq)
    omics_np = np.load(args.omics)

    model = MultiModalSeq2HiCModel(num_omics_features=4)
    st = torch.load(str(WEIGHTS), map_location='cpu')
    if all(k.startswith('module.') for k in st.keys()):
        st = {k.replace('module.', ''): v for k, v in st.items()}
    model.load_state_dict(st)
    model.eval()

    seq_t = torch.from_numpy(seq_np)
    omics_t = torch.from_numpy(omics_np)
    n_steps = 50
    grad_sum = None
    for i in range(args.lo, args.hi):
        alpha = i / n_steps
        x = (seq_t * alpha).unsqueeze(0)
        x.requires_grad_(True)
        out = model(x, omics_t.unsqueeze(0))[0, args.ti, args.tj]
        grad = torch.autograd.grad(out, x)[0].squeeze(0).numpy()
        grad_sum = grad if grad_sum is None else grad_sum + grad
        if (i - args.lo) % 5 == 4:
            print(f'[worker{args.proc}] step {i+1}/{args.hi} done', flush=True)
    np.save(args.out, grad_sum)
    print(f'[worker{args.proc}] gradient part saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
