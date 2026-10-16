"""Erasure probe: run one model on every impossible image and possible control as it is and with each junction
erased (pdepth/probe.py), and store the loop profile of the visible cubes (erased cubes are NaN).

usage: python scripts/run_probe.py <model-key> [stim_glob] [n_seeds]     (PROBE_TAG=_loops for data/loops)
writes results/probe<PROBE_TAG>_<model>.csv (one row per image, condition and seed) and, for layout 90 (FIG below),
outputs/probe/<model>/<stim>__<cond>.npy (float16 maps, seed 0)
"""
import os, sys, glob, csv, time, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.models import REGISTRY
from pdepth.readout import cube_masks, cube_profile
from pdepth.probe import erase, junction_cubes

META = ["set", "angle", "perm", "tex", "mirror", "possible", "tau", "nuis", "style"]
FIG = {"A_a090_p0_t0", "B_a090_k0", "B_a090_k1", "B_a090_k2"}
if len(sys.argv) < 2:
    sys.exit(__doc__)
key = sys.argv[1]
stims = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else "data/stim2/[AB]_*.npz"))
if not stims:
    sys.exit("no stimuli found: run scripts/make_stimuli.py first (results/ already holds this table)")
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.set_num_threads(os.cpu_count())
out_dir = f"outputs/probe/{key}"
os.makedirs(out_dir, exist_ok=True); os.makedirs("results", exist_ok=True)
model = REGISTRY[key]()
rows, t0 = [], time.time()
for i, f in enumerate(stims):
    name = os.path.basename(f)[:-4]
    st = dict(np.load(f))
    n_cubes, n_bars = int(st["n_cubes"]), int(st["n_bars"]) if "n_bars" in st else 3
    masks = cube_masks(st["cube"], n_cubes)
    conds = [("none", st["img"], None)] + [(f"e{j}",) + erase(st, junction_cubes(j, n_cubes)) for j in range(n_bars)]
    for cond, img, hole in conds:
        keep = masks if hole is None else [m & ~hole for m in masks]
        for seed in range(n_seeds):
            depth = model(img, seed=seed)
            if seed == 0 and name in FIG:
                np.save(f"{out_dir}/{name}__{cond}.npy", depth.astype(np.float16))
            prof = cube_profile(depth, keep)
            rows.append(dict(model=key, stim=name, cond=cond, seed=seed, **{k: (st[k].item() if k in st else "") for k in META},
                             n_bars=n_bars, **{f"prof{j}": v for j, v in enumerate(prof)}))
    if (i + 1) % 25 == 0:
        print(f"{key}: {i + 1}/{len(stims)} ({time.time() - t0:.0f}s)", flush=True)
tag = os.environ.get("PROBE_TAG", "")
with open(f"results/probe{tag}_{key}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
    w.writeheader(); w.writerows(rows)
print(f"done {key}: {len(rows)} rows ({time.time() - t0:.0f}s)")
