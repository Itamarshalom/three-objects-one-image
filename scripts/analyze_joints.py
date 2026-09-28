"""Are joints read locally? Compare the joint jumps a network produces on the impossible image with those it
produces on the possible control of object k at the same layout and chirality (set A with perm 0 / texture 0 at
the 15-deg layouts, vs set B). The two images differ only at joint k (stub visible or not); a network that reads
each joint from its local appearance should give the same jumps at the two shared joints (j != k) and, at joint k,
its ordinary real-joint step r_k instead of the control's tear g_k.
Everything is in units of the excess-tear scale of scripts/analyze_excess.py: |difference| / (g_j - r_j), with
g_j, r_j from the network's own three controls at that layout. For the samplers, jumps are averaged over noise seeds.
Null for the shared joints: the same comparison against the control of object k at a layout 45 deg away.
usage: python scripts/analyze_joints.py   -> results/summary_joints.json"""
import os, sys, json, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load, boot_ci
from pdepth.loop import bars_and_joints

P = [f"prof{j}" for j in range(15)]
RUNS = {m: f"results/readout_{m}.csv" for m in ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft"]}
RUNS["mg11-1"] = "results/mg11_1step.csv"; RUNS["mg10-1"] = "results/mg10_1step.csv"


def seed_mean_jumps(rows):
    acc = defaultdict(list)
    for r in rows:
        if r["set"] in "AB" and r.get("seed", 0) != -1:
            acc[r["stim"]].append(bars_and_joints(np.array([r[p] for p in P], float))[1])
    return {s: np.mean(v, 0) for s, v in acc.items()}


if __name__ == "__main__":
    S = {}
    for m, fn in RUNS.items():
        if not os.path.exists(fn):
            continue
        J = seed_mean_jumps(load(fn))
        shared, vs_tear, vs_real, null, lay = [], [], [], [], []
        for a in range(0, 120, 15):
            for mi in ("", "_m"):
                sa, sb = f"A_a{a:03d}_p0_t0{mi}", [f"B_a{a:03d}_k{k}{mi}" for k in range(3)]
                if sa not in J or any(b not in J for b in sb):
                    continue
                C = [J[b] for b in sb]
                g = np.array([C[j][j] for j in range(3)])
                r = np.array([np.mean([C[k][j] for k in range(3) if k != j]) for j in range(3)])
                u = g - r                                       # one tear in excess-tear units, per joint
                a2 = (a + 45) % 120
                sb2 = [f"B_a{a2:03d}_k{k}{mi}" for k in range(3)]
                for k in range(3):
                    o = [j for j in range(3) if j != k]
                    shared.append(np.mean(np.abs(J[sa][o] - C[k][o]) / u[o]))
                    vs_tear.append(abs(J[sa][k] - g[k]) / u[k])
                    vs_real.append(abs(J[sa][k] - r[k]) / u[k])
                    lay.append(a)
                    if sb2[k] in J:
                        null.append(np.mean(np.abs(J[sa][o] - J[sb2[k]][o]) / u[o]))
        if not shared:
            continue
        ci = lambda x: boot_ci(np.array(x), np.median, groups=lay[:len(x)] if len(x) == len(lay) else None)
        S[m] = dict(shared=ci(shared), vs_tear=ci(vs_tear), vs_real=ci(vs_real), null_shared=ci(null), n=len(shared))
        d = S[m]
        print(f"{m:7s} shared joints {d['shared'][0]:.2f} [{d['shared'][1]:.2f},{d['shared'][2]:.2f}]  (null {d['null_shared'][0]:.2f})"
              f"   joint k vs control's tear {d['vs_tear'][0]:.2f}   vs ordinary step {d['vs_real'][0]:.2f}   n={d['n']}")
    json.dump(S, open("results/summary_joints.json", "w"), indent=1, default=float)
