"""Bar slopes on the impossible image, normalised by the network's own controls (a check the report quotes as
inconclusive).

A mixture of the three objects and their depth reversals (the posterior mean of a belief over upright and reversed
objects, any weights) has three bar slopes c * s with one common factor c = P(up) - P(rev). In a network's own units we
normalise each bar's slope by the mean slope of that bar on the network's three possible controls at the same layout
(seed-averaged for the samplers): such a mixture would give three EQUAL normalised slopes. We report the share of
impossible outputs whose normalised slopes have mixed signs, and the median spread max - min, next to the same numbers
for each control normalised by the other two controls of its layout. On our renders the controls themselves have
mixed-sign normalised slopes in 19-81% of cases (the networks read bar slopes much less consistently than joint
steps), so this test cannot separate the two accounts except for the networks where the impossible image is far above
its control floor.
usage: python scripts/analyze_bars.py   -> results/summary_bars.json"""
import os, sys, json, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load, boot_ci
from pdepth.loop import bars_and_joints

P = [f"prof{j}" for j in range(15)]
RUNS = {m: f"results/readout_{m}.csv" for m in ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft"]}
RUNS["mg11-1"] = "results/mg11_1step.csv"; RUNS["mg10-1"] = "results/mg10_1step.csv"


def lay(r):
    return (int(r["angle"]) - int(r["angle"]) % 15, bool(r["mirror"]))


def stats(s):
    s = np.asarray(s)
    mixed = (s.min(1) < 0) & (s.max(1) > 0)
    return mixed, s.max(1) - s.min(1)


if __name__ == "__main__":
    S = {}
    for m, fn in RUNS.items():
        if not os.path.exists(fn):
            continue
        rows = [r for r in load(fn) if r.get("seed", 0) != -1]
        ctrl = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["set"] == "B":
                ctrl[lay(r)][int(r["possible"])].append(bars_and_joints(np.array([r[p] for p in P], float))[0])
        ref = {key: {k: np.mean(v, 0) for k, v in d.items()} for key, d in ctrl.items() if len(d) == 3}
        imp, g_imp, con = [], [], []
        for r in rows:
            if lay(r) not in ref:
                continue
            b = bars_and_joints(np.array([r[p] for p in P], float))[0]
            if r["set"] == "A":
                imp.append(b / np.mean(list(ref[lay(r)].values()), 0)); g_imp.append(lay(r)[0])
            elif r["set"] == "B":
                k = int(r["possible"])
                con.append(b / np.mean([v for kk, v in ref[lay(r)].items() if kk != k], 0))
        mi, si = stats(imp); mc, sc = stats(con)
        S[m] = dict(n=len(imp), mixed=boot_ci(mi.astype(float), np.mean, groups=g_imp), spread=float(np.median(si)),
                    ctrl_mixed=float(mc.mean()), ctrl_spread=float(np.median(sc)),
                    reversed_mean=float(np.mean(np.mean(imp, 1) < 0)))
        d = S[m]
        print(f"{m:7s} impossible: mixed-sign bars {d['mixed'][0]:.2f} [{d['mixed'][1]:.2f},{d['mixed'][2]:.2f}]  spread {d['spread']:.2f}"
              f"   | controls: mixed {d['ctrl_mixed']:.2f}  spread {d['ctrl_spread']:.2f}   (n={d['n']})")
    json.dump(S, open("results/summary_bars.json", "w"), indent=1, default=float)
