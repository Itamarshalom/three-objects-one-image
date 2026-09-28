"""Marigold-family per-seed samples on a set of stimuli, for several DDIM step counts (seeds batched).

Writes one read-out row per (stimulus, steps, seed) to results/<tag>.csv, one row for the seed-mean map
per (stimulus, steps) to results/<tag>_seedmean.csv, maps of the first 8 seeds and the seed mean to
outputs/<tag>/<stim>.npz, and with --traj the read-out of the x0-prediction at every step
(results/<tag>_traj.csv).

usage: python scripts/run_marigold.py <tag> "<stim_glob>" <steps,...> <n_seeds> [--traj] [--model <hf id>]
"""
import os, sys, glob, csv, time, argparse, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.models import Marigold
from pdepth.readout import readout, cube_masks

META = ["set", "angle", "perm", "tex", "mirror", "possible", "tau", "nuis", "style"]
ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("stims"); ap.add_argument("steps"); ap.add_argument("n_seeds", type=int)
ap.add_argument("--traj", action="store_true"); ap.add_argument("--model", default="prs-eth/marigold-depth-v1-1")
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--res", type=int, default=768)
a = ap.parse_args()
steps_list = [int(s) for s in a.steps.split(",")]
seeds = list(range(a.n_seeds))
files = sorted(glob.glob(a.stims))
if not files:
    sys.exit("no stimuli found: run scripts/make_stimuli.py and the queue's sel_* lines first "
             "(results/ already holds this table)")
os.makedirs("results", exist_ok=True); os.makedirs(f"outputs/{a.tag}", exist_ok=True)
m = Marigold(a.model, res=a.res)


def row(name, st, steps, seed, r, extra=None):
    d = dict(stim=name, **{k: (st[k].item() if k in st else "") for k in META}, steps=steps, seed=seed,
             w1=r["w"][0], w2=r["w"][1], w3=r["w"][2], scale=r["scale"], r2=r["r2"], commit=r["commit"],
             zre=r["z"].real, zim=r["z"].imag, corner=r["corner"],
             **{f"slope{j}": v for j, v in enumerate(r["bar_slope"])},
             **{f"prof{j}": v for j, v in enumerate(r["profile"])})
    if extra:
        d.update(extra)
    return d


def dump(fn, rr):
    if rr:
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)


rows, mean_rows, traj_rows = [], [], []
t0 = time.time()
for fi, f in enumerate(files):
    name = os.path.basename(f)[:-4]
    st = dict(np.load(f))
    masks = cube_masks(st["cube"], int(st["n_cubes"]))
    maps = {}
    for steps in steps_list:
        if a.traj:
            fin, traj = m.sample(st["img"], seeds, steps, record_x0=True, batch=a.batch)
            for i, s in enumerate(seeds):
                for k in range(steps):
                    traj_rows.append(row(name, st, steps, s, readout(traj[i, k], st, masks), dict(step=k + 1)))
        else:
            fin = m.sample(st["img"], seeds, steps, batch=a.batch)
        for i, s in enumerate(seeds):
            rows.append(row(name, st, steps, s, readout(fin[i], st, masks)))
        mean_rows.append(row(name, st, steps, -1, readout(fin.mean(0), st, masks)))
        maps[f"s{steps}"] = fin[:8].astype(np.float16)
        maps[f"s{steps}_mean"] = fin.mean(0).astype(np.float16)
    np.savez_compressed(f"outputs/{a.tag}/{name}.npz", **maps)
    print(f"{fi + 1}/{len(files)} {name} ({time.time() - t0:.0f}s)", flush=True)
    if (fi + 1) % 5 == 0 or fi + 1 == len(files):              # write as we go
        dump(f"results/{a.tag}.csv", rows); dump(f"results/{a.tag}_seedmean.csv", mean_rows)
        dump(f"results/{a.tag}_traj.csv", traj_rows)
print("done")
