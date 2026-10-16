"""Render random POSSIBLE cube scenes (pdepth/scenes.py) for training read-outs and heads.

usage: python scripts/make_scenes.py <out_dir> <n_scenes> [n_workers] [first_seed]
writes <out_dir>/scene_<seed>.npz (img, depth, grad, disc, fg, cube, face, px, mirror, n_cubes); seeds are consecutive
"""
import os, sys, numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.scenes import sample

if len(sys.argv) < 3:
    sys.exit(__doc__)
OUT, N = sys.argv[1], int(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
FIRST = int(sys.argv[4]) if len(sys.argv) > 4 else 0


def run(seed):
    fn = os.path.join(OUT, f"scene_{seed:05d}.npz")
    if not os.path.exists(fn):
        np.savez_compressed(fn, **sample(seed))
    return seed


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool(WORKERS) as p:
        for i, s in enumerate(p.imap_unordered(run, range(FIRST, FIRST + N)), 1):
            if i % 100 == 0 or i == N:
                print(f"{i}/{N}", flush=True)
