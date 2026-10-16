"""Run one normal network (pdepth/normals.py) on every impossible image and possible control.

For each image and each of the 8 axis-sign conventions s (applied as n * s), stores the median cosine with the exact
normals over face interiors and the along-bar slope of each bar's interior cubes (change of depth per unit length,
the objects' value is -1/sqrt(2)); scripts/analyze_normals.py picks one convention per network on the controls.
usage: python scripts/run_normals.py <model-key> [stim_glob] [n_seeds]
writes results/normals_<model>.csv and, for layout 90 (FIG below), outputs/normals/<model>/<stim>.npy (float16)
"""
import os, sys, glob, csv, time, itertools, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.normals import NORMALS, gt_normals, bar_directions, along_bar_slope
from pdepth.readout import cube_masks

META = ["set", "angle", "perm", "tex", "mirror", "possible"]
FIG = {"A_a090_p0_t0", "B_a090_k0", "B_a090_k1", "B_a090_k2"}
SIGNS = list(itertools.product((1, -1), repeat=3))
if len(sys.argv) < 2:
    sys.exit(__doc__)
key = sys.argv[1]
stims = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else "data/stim2/[AB]_*.npz"))
if not stims:
    sys.exit("no stimuli found: run scripts/make_stimuli.py first (results/ already holds this table)")
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.set_num_threads(os.cpu_count())
out_dir = f"outputs/normals/{key}"
os.makedirs(out_dir, exist_ok=True); os.makedirs("results", exist_ok=True)
model = NORMALS[key]()
rows, t0 = [], time.time()
for i, f in enumerate(stims):
    name = os.path.basename(f)[:-4]
    st = dict(np.load(f))
    gt, ok = gt_normals(st)
    masks = cube_masks(st["cube"], int(st["n_cubes"]))
    bars = [np.any([masks[c] for c in range(5 * j + 1, 5 * j + 5)], 0) & ok for j in range(3)]
    t = bar_directions(st)
    for seed in range(n_seeds):
        n = model(st["img"], seed=seed)
        if seed == 0 and name in FIG:
            np.save(f"{out_dir}/{name}.npy", n.astype(np.float16))
        row = dict(model=key, stim=name, seed=seed, **{k: (st[k].item() if k in st else "") for k in META})
        for s in SIGNS:
            ns = n * np.array(s)
            tag = "".join("+" if v > 0 else "-" for v in s)
            row[f"cos{tag}"] = float(np.median((ns[ok] * gt[ok]).sum(-1)))
            for j in range(3):
                row[f"slope{j}{tag}"] = along_bar_slope(ns, bars[j], t[j])
        rows.append(row)
    if (i + 1) % 25 == 0:
        print(f"{key}: {i + 1}/{len(stims)} ({time.time() - t0:.0f}s)", flush=True)
with open(f"results/normals_{key}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"done {key}: {len(rows)} rows ({time.time() - t0:.0f}s)")
