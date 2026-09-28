"""Excess tear at each joint, in units of the network's own tear (primary test of the theory).

From the three possible controls of a layout (set B, same chirality, same 15-degree layout bin: the control at the
same angle or the nearest below) we get, for every joint j, the step a
network makes at j when j is a REAL joint (controls k != j, averaged) and when j carries the TEAR (control k = j).
For any output at that layout:   e_j = (J_j - real_j) / (tear_j - real_j).
The mean of the three objects (any weights p) and each single object pay the whole
mismatch at the joints: sum_j e_j = 1 for the mean (e = p) and for a single object (e one-hot); a belief that also
includes depth-reversed objects gives P(up) - P(rev) (see docs/theory.md). A network that makes its ordinary
real-joint step at every joint gives e = 0. Samplers (Lotus-G, Marigold): all 32 noise seeds are pooled, each seed
one sample; the intervals resample whole 15-degree layout bins.
usage: python scripts/analyze_excess.py   -> prints, results/summary_excess.json, results/excess_rows.csv"""
import os, sys, csv, json, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load, boot_ci
from pdepth.loop import bars_and_joints

P = [f"prof{j}" for j in range(15)]
RUNS = {m: f"results/readout_{m}.csv" for m in ["dav2-s", "dav2-b", "dav2-l", "lotus-d", "lotus-g", "e2eft", "dpt-l", "zoe"]}
RUNS.update({"mg11-1": "results/mg11_1step.csv", "mg10-1": "results/mg10_1step.csv"})


def lay(r):
    return (int(r["angle"]) - int(r["angle"]) % 15, bool(r["mirror"]))


def references(rows):
    """per (layout, chirality): real_j and tear_j for j = 0, 1, 2 from the three possible controls
    (for the samplers, the controls' jumps are averaged over noise seeds first)"""
    ctrl = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["set"] == "B":
            ctrl[lay(r)][int(r["possible"])].append(bars_and_joints(np.array([r[p] for p in P], float))[1])
    ref = {}
    for key, d in ctrl.items():
        if len(d) < 3:
            continue
        d = {k: np.mean(v, 0) for k, v in d.items()}
        real = np.array([np.mean([d[k][j] for k in range(3) if k != j]) for j in range(3)])
        tear = np.array([d[j][j] for j in range(3)])
        ref[key] = (real, tear)
    return ref


def excess(rows):
    ref = references(rows)
    out = []
    for r in rows:
        if lay(r) not in ref:
            continue
        real, tear = ref[lay(r)]
        J = bars_and_joints(np.array([r[p] for p in P], float))[1]
        out.append(dict(r=r, e=(J - real) / (tear - real)))
    return out


if __name__ == "__main__":
    S, rows_out = {}, []
    for m, fn in RUNS.items():
        if not os.path.exists(fn):
            continue
        rows = [r for r in load(fn) if r.get("seed", 0) != -1]          # samplers: every noise seed is one sample
        out = excess(rows)
        S[m] = {}
        for st in ("B", "A", "D", "E"):
            O = [o for o in out if o["r"]["set"] == st]
            if not O:
                continue
            tot = np.array([o["e"].sum() for o in O]); g = np.array([o["r"]["angle"] - o["r"]["angle"] % 15 for o in O])
            mx = np.array([o["e"].max() for o in O])
            S[m][st] = dict(n=len(O), sum_e=boot_ci(tot, np.median, groups=g), max_e=boot_ci(mx, np.median, groups=g))
            d = S[m][st]
            print(f"{m:7s} set {st}: sum of excess tears {d['sum_e'][0]:+.2f} [{d['sum_e'][1]:+.2f},{d['sum_e'][2]:+.2f}]"
                  f"   largest single-joint excess {d['max_e'][0]:+.2f}   (n={d['n']})")
        C = [o for o in out if o["r"]["set"] == "C"]
        if C:
            curve = {}
            for t in sorted({o["r"]["tau"] for o in C}):
                ek = [o["e"][int(o["r"]["possible"])] for o in C if o["r"]["tau"] == t]
                curve[f"{t:.2f}"] = float(np.median(ek))
            S[m]["C_ek"] = curve
            print(f"{m:7s} morph: excess at the revealed joint vs tau:", {k: round(v, 2) for k, v in curve.items()})
        for o in out:
            rows_out.append(dict(model=m, stim=o["r"]["stim"], seed=o["r"].get("seed", ""), set=o["r"]["set"], angle=o["r"]["angle"],
                                 mirror=o["r"]["mirror"], possible=o["r"].get("possible", ""), tau=o["r"].get("tau", ""),
                                 e1=o["e"][0], e2=o["e"][1], e3=o["e"][2]))
    json.dump(S, open("results/summary_excess.json", "w"), indent=1, default=float)
    with open("results/excess_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
