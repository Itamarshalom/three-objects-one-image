"""Ground-truth loop profiles of the three objects for every stimulus (per-cube medians over the same eroded
masks the read-out uses); scripts/paper_numbers.py reads it for the read-out's accuracy on the ground truth.
usage: python scripts/gt_profiles.py [stim_glob]   -> results/gt_profiles.csv (stim, k, prof0..prof14)"""
import os, sys, glob, csv, numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.readout import cube_masks, cube_profile


def one(f):
    st = dict(np.load(f)); name = os.path.basename(f)[:-4]
    masks = cube_masks(st["cube"], int(st["n_cubes"]))
    return [dict(stim=name, k=k, **{f"prof{j}": v for j, v in enumerate(cube_profile(st["depth"][k], masks))})
            for k in range(3)]


if __name__ == "__main__":
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "data/stim2/*.npz"))
    with Pool(16) as p:
        rows = [r for rr in p.map(one, files) for r in rr]
    if not rows:
        sys.exit("no stimuli found: run scripts/make_stimuli.py first (results/ already holds this table)")
    with open("results/gt_profiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "rows")
