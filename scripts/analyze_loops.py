"""Penrose loop vs closed possible loops (data/loops: skew hexagon H, rhombus R), intact and with one junction erased.

Scale-free measures, from each output's own loop read-out (pdepth/loop.py):
  bars right  : share of bars whose slope has the sign of the exact local slope (the objects' for the Penrose loop)
  released    : (J_j with j erased - J_j intact) / mean |bar change| of the erased output, at the erased junction j.
                The Penrose loop's obstruction is three bar changes (T = 3 x T/3), so a network that reads every bar and
                junction locally releases about 3 there; a closed possible loop has no obstruction and should release ~0.
usage: python scripts/analyze_loops.py [model ...]   (reads results/probe_<model>.csv and results/probe_loops_<model>.csv)"""
import os, sys, json, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load
from pdepth.loop import bars_and_joints, N_BAR

SIGN_PENROSE = -1.0                          # every bar of every object comes nearer along the loop


def ring_signs(st_path):
    d = np.load(st_path)
    nb = int(d["n_bars"]); cube = d["cube"]; dep = d["depth"][0]
    prof = np.array([np.nanmedian(dep[cube == c]) for c in range(int(d["n_cubes"]))])
    return np.sign(bars_and_joints(prof, nb)[0])


def summarise(rows, loops_dir="data/loops"):
    by = defaultdict(dict)
    for r in rows:
        nb = int(r.get("n_bars") or 3)
        prof = np.array([r.get(f"prof{j}", np.nan) for j in range(N_BAR * nb)], float)
        by[(r["stim"], r.get("seed", 0))][r["cond"]] = bars_and_joints(prof, nb)
    out = defaultdict(lambda: defaultdict(list))
    signs = {}
    for (stim, seed), d in by.items():
        if "none" not in d:
            continue
        fam = stim[0]
        if fam in "HR":
            if stim not in signs:
                signs[stim] = ring_signs(os.path.join(loops_dir, stim + ".npz"))
            sg = signs[stim]
        elif fam == "A":
            sg = np.full(3, SIGN_PENROSE)
        else:
            continue
        b0, J0 = d["none"]
        out[fam]["bars_right_intact"].append(np.mean(np.sign(b0) == sg))
        for c, (b, J) in d.items():
            if c == "none":
                continue
            j = int(c[1:])
            scale = np.nanmean(np.abs(N_BAR * b))
            out[fam]["released"].append((J[j] - J0[j]) / scale)
            out[fam]["bars_right_erased"].append(np.nanmean(np.sign(b[np.isfinite(b)]) == sg[np.isfinite(b)]))
    return {f: {k: float(np.median(v)) if k == "released" else float(np.mean(v)) for k, v in m.items()}
            | {"n": len(m["released"]), "released_iqr": [float(np.percentile(m["released"], 25)),
                                                         float(np.percentile(m["released"], 75))]}
            for f, m in out.items()}


if __name__ == "__main__":
    S = {}
    for m in sys.argv[1:] or ["dav2-s"]:
        rows = []
        for fn in (f"results/probe_{m}.csv", f"results/probe_loops_{m}.csv"):
            if os.path.exists(fn):
                rows += [r for r in load(fn) if r.get("set", "A") in ("A", "H", "R") or str(r["stim"])[0] in "AHR"]
        S[m] = summarise(rows)
        for fam, v in sorted(S[m].items()):
            name = {"A": "Penrose", "H": "hexagon", "R": "rhombus"}[fam]
            print(f"{m:8s} {name:8s} bars right: intact {v['bars_right_intact']:.2f}, one junction erased "
                  f"{v['bars_right_erased']:.2f} | released at the erased junction {v['released']:+.2f} "
                  f"[{v['released_iqr'][0]:+.2f},{v['released_iqr'][1]:+.2f}] bar changes (n={v['n']})")
    json.dump(S, open("results/summary_loops.json", "w"), indent=1)
