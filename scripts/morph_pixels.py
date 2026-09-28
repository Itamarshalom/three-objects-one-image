"""How many pixels does each gap-morph step change? Counts pixels where the tau image differs from the tau = 0
(impossible) image of the same layout, for all 12 layout x object cells. -> results/morph_pixels.csv"""
import os, sys, glob, csv, numpy as np

rows = []
for f in sorted(glob.glob("data/stim2/C_*_tau000.npz")):
    base = np.load(f)["img"].astype(int)
    stem = f.replace("_tau000.npz", "")
    for g in sorted(glob.glob(stem + "_tau*.npz")):
        tau = int(g[-7:-4]) / 100
        d = np.abs(np.load(g)["img"].astype(int) - base).sum(-1)
        rows.append(dict(cell=os.path.basename(stem), tau=tau, changed_px=int((d > 0).sum()), changed_px_strong=int((d > 30).sum())))
if not rows:
    sys.exit("no stimuli found: run scripts/make_stimuli.py first (results/ already holds this table)")
with open("results/morph_pixels.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for t in sorted({r["tau"] for r in rows}):
    v = [r["changed_px"] for r in rows if r["tau"] == t]
    print(f"tau={t:.2f}: changed pixels median {np.median(v):.0f} (range {min(v)}-{max(v)}), share of image {np.median(v)/512**2*100:.3f}%")
