"""Read-out sensitivity: recompute loop profiles from the saved depth maps with different mask erosions and per-cube
statistics, then the total excess tear on the impossible image (set A) and the control readability.
usage (where outputs/pred/<model>/*.npy and data/stim2 exist): python scripts/sensitivity.py <model> ...
-> results/sensitivity.csv"""
import os, sys, glob, csv, numpy as np
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.readout import cube_masks
from pdepth.loop import bars_and_joints

VARIANTS = [(1, "median"), (3, "median"), (5, "median"), (3, "mean")]


def profiles(args):
    model, f = args
    st = np.load(f); name = os.path.basename(f)[:-4]
    d = np.load(f"outputs/pred/{model}/{name}.npy").astype(float)
    out = {}
    for er, stat in VARIANTS:
        masks = cube_masks(st["cube"], int(st["n_cubes"]), erode=er)
        fn = np.nanmedian if stat == "median" else np.nanmean
        out[(er, stat)] = np.array([fn(d[m]) for m in masks])
    return name, str(st["set"]), int(st["angle"]), bool(st["mirror"]), int(st["possible"]), out


def lay(a, m):
    return (a - a % 15, m)


if __name__ == "__main__":
    rows = []
    files = sorted(glob.glob("data/stim2/A_*.npz")) + sorted(glob.glob("data/stim2/B_*.npz"))
    if len(sys.argv) < 2 or not files:
        sys.exit("needs model keys, data/stim2 and outputs/pred (results/ already holds this table)")
    for model in sys.argv[1:]:
        with Pool(16) as p:
            res = p.map(profiles, [(model, f) for f in files])
        for var in VARIANTS:
            J = {r[0]: (r[1], r[2], r[3], r[4], bars_and_joints(r[5][var])[1]) for r in res}
            ctrl = defaultdict(dict)
            for name, (s, a, m, k, j) in J.items():
                if s == "B":
                    ctrl[lay(a, m)][k] = j
            ref = {}
            for key, d in ctrl.items():
                if len(d) == 3:
                    ref[key] = (np.array([np.mean([d[k][j] for k in range(3) if k != j]) for j in range(3)]),
                                np.array([d[j][j] for j in range(3)]))
            sums = [((j - ref[lay(a, m)][0]) / (ref[lay(a, m)][1] - ref[lay(a, m)][0])).sum()
                    for name, (s, a, m, k, j) in J.items() if s == "A" and lay(a, m) in ref]
            readable = np.mean([int(np.argmax(j)) == k for name, (s, a, m, k, j) in J.items() if s == "B"])
            rows.append(dict(model=model, erode=var[0], stat=var[1], sum_e_median=float(np.median(sums)),
                             sum_e_iqr_lo=float(np.percentile(sums, 25)), sum_e_iqr_hi=float(np.percentile(sums, 75)),
                             control_tear_at_gap=float(readable)))
            print(rows[-1], flush=True)
    with open("results/sensitivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
