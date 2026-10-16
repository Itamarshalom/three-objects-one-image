"""Run one model on every stimulus (optionally several seeds); save maps (float16) and one read-out row
per (image, seed).

usage: python scripts/run_models.py <model-key> [stim_glob] [n_seeds]
writes outputs/pred/<model>/<stim>.npy (the seed-0 map, float16) and results/readout_<model>.csv
"""
import os, sys, glob, csv, time, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.models import REGISTRY
from pdepth.readout import readout, cube_masks

META = ["set", "angle", "perm", "tex", "mirror", "possible", "tau", "nuis", "style"]
if len(sys.argv) < 2:
    sys.exit(__doc__)
key = sys.argv[1]
stims = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else "data/stim2/*.npz"))
if not stims:
    sys.exit("no stimuli found: run scripts/make_stimuli.py first (results/ already holds this table)")
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.set_num_threads(os.cpu_count())
out_dir = f"outputs/pred/{key}"
os.makedirs(out_dir, exist_ok=True); os.makedirs("results", exist_ok=True)
model = REGISTRY[key]()
rows, t0 = [], time.time()
for i, f in enumerate(stims):
    name = os.path.basename(f)[:-4]
    st = dict(np.load(f))
    masks = cube_masks(st["cube"], int(st["n_cubes"]))
    for seed in range(n_seeds):
        depth = model(st["img"], seed=seed)
        if seed == 0:
            np.save(f"{out_dir}/{name}.npy", depth.astype(np.float16))
        r = readout(depth, st, masks)
        row = dict(model=key, stim=name, seed=seed, **{k: (st[k].item() if k in st else "") for k in META})
        row.update(w1=r["w"][0], w2=r["w"][1], w3=r["w"][2], scale=r["scale"], r2=r["r2"], commit=r["commit"],
                   zre=r["z"].real, zim=r["z"].imag, corner=r["corner"],
                   **{f"slope{j}": v for j, v in enumerate(r["bar_slope"])},
                   **{f"prof{j}": v for j, v in enumerate(r["profile"])})
        rows.append(row)
    if (i + 1) % 50 == 0:
        print(f"{key}: {i + 1}/{len(stims)} ({time.time() - t0:.0f}s)", flush=True)
with open(f"results/readout_{key}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"done {key}: {len(rows)} rows ({time.time() - t0:.0f}s)")
