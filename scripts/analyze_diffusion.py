"""Excess tear for the sampling networks, per seed (see scripts/analyze_excess.py for e_j).
References (real-joint step r_j and gap step g_j per layout) come from the SAME run configuration: the possible
controls in the same file (one-step runs) or in the matching *_refs file (multi-step runs), averaged over seeds;
for the multi-step gap morph, the tau = 1 images of the morph run itself (which are the controls) replace them
at its layouts (0 and 60 deg). The multi-step references exist at 0/30/60/90 deg only, so the multi-step
samples at 15/45/75/105 deg in *_multi.csv are skipped.
Reports, per configuration: median total excess over images x seeds, the typical seed-to-seed SD of the total,
and the share of samples that pay at least half a tear (sum e >= 0.5).
usage: python scripts/analyze_diffusion.py   -> results/summary_diffusion.json"""
import os, sys, json, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load, boot_ci
from pdepth.loop import bars_and_joints

P = [f"prof{j}" for j in range(15)]


def lay(r):
    return (int(r["angle"]) - int(r["angle"]) % 15, bool(r["mirror"]))


def refs_from(rows, steps=None):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if steps is not None and r.get("steps") != steps:
            continue
        is_ctrl = r["set"] == "B" or (r["set"] == "C" and abs(float(r["tau"]) - 1.0) < 1e-6)
        if is_ctrl and r.get("seed", 0) != -1:
            acc[lay(r)][int(r["possible"])].append(bars_and_joints(np.array([r[p] for p in P], float))[1])
    ref = {}
    for key, d in acc.items():
        if len(d) < 3:
            continue
        m = {k: np.mean(v, 0) for k, v in d.items()}
        ref[key] = (np.array([np.mean([m[k][j] for k in range(3) if k != j]) for j in range(3)]),
                    np.array([m[j][j] for j in range(3)]))
    return ref


def excess(rows, ref, sets=("A",), steps=None):
    by = defaultdict(list)
    for r in rows:
        if r["set"] not in sets or r.get("seed", 0) == -1 or (steps is not None and r.get("steps") != steps):
            continue
        if lay(r) not in ref:
            continue
        real, gap = ref[lay(r)]
        e = (bars_and_joints(np.array([r[p] for p in P], float))[1] - real) / (gap - real)
        by[r["stim"]].append((r, e))
    return by


def summarise(by):
    tot = [e.sum() for v in by.values() for _, e in v]
    grp = [s for s, v in by.items() for _ in v]
    sd = [np.std([e.sum() for _, e in v]) for v in by.values() if len(v) > 1]
    tot = np.array(tot)
    return dict(n_images=len(by), n_samples=len(tot), sum_e=boot_ci(tot, np.median, groups=grp),
                seed_sd=float(np.median(sd)) if sd else None, share_half_tear=float(np.mean(tot >= 0.5)))


if __name__ == "__main__":
    S = {}
    one = {"lotus-g": "results/readout_lotus-g.csv", "mg10-1": "results/mg10_1step.csv", "mg11-1": "results/mg11_1step.csv"}
    for m, f in one.items():
        if os.path.exists(f):
            rows = load(f)
            S[m] = summarise(excess(rows, refs_from(rows)))
    multi = [("mg11", 4, "results/mg11_multi.csv", "results/mg11_multi_refs.csv"),
             ("mg11", 10, "results/mg11_multi.csv", "results/mg11_multi_refs.csv"),
             ("mg10", 10, "results/mg10_multi.csv", "results/mg10_multi_refs.csv")]
    for m, st, f, fr in multi:
        if os.path.exists(f) and os.path.exists(fr):
            ref = refs_from(load(fr), st)
            S[f"{m}-{st}"] = summarise(excess(load(f), ref, steps=st))
    for k, d in S.items():
        v, lo, hi = d["sum_e"]
        sd = "n/a" if d["seed_sd"] is None else f"{d['seed_sd']:.2f}"
        print(f"{k:9s} total excess {v:+.2f} [{lo:+.2f},{hi:+.2f}]  seed SD {sd}  "
              f"samples paying >= half a tear {d['share_half_tear']:.2f}  ({d['n_images']} images, {d['n_samples']} samples)")
    # gap morph under multi-step sampling
    for m, st, f, fr in [("mg11", 4, "results/mg11_multi.csv", "results/mg11_multi_refs.csv"),
                         ("mg11", 10, "results/mg11_multi.csv", "results/mg11_multi_refs.csv"),
                         ("mg10", 10, "results/mg10_multi_morph.csv", "results/mg10_multi_refs.csv")]:
        if not (os.path.exists(f) and os.path.exists(fr)):
            continue
        rows = load(f)
        ref = refs_from(load(fr), st)
        ref.update(refs_from(rows, st))
        by = excess(rows, ref, sets=("C",), steps=st)
        curve = defaultdict(list)
        for v in by.values():
            for r, e in v:
                curve[float(r["tau"])].append(e[int(r["possible"])])
        S[f"{m}-{st}-morph"] = {f"{t:.2f}": float(np.median(v)) for t, v in sorted(curve.items())}
        print(f"{m}-{st} morph:", {k: round(v, 2) for k, v in S[f"{m}-{st}-morph"].items()})
    # x0-prediction along 10-step trajectories: does any sample start paying a tear on the way?
    # (references: the same network's final 10-step samples of the possible controls)
    for m, f, fr in [("mg11", "results/mg11_traj_traj.csv", "results/mg11_multi_refs.csv"),
                     ("mg10", "results/mg10_traj_traj.csv", "results/mg10_multi_refs.csv")]:
        if not (os.path.exists(f) and os.path.exists(fr)):
            continue
        ref, rows = refs_from(load(fr), 10), load(f)
        out = {}
        for k in sorted({r["step"] for r in rows}):
            by = excess([r for r in rows if r["step"] == k], ref, steps=10)
            tot = np.array([e.sum() for v in by.values() for _, e in v])
            mx = np.array([e.max() for v in by.values() for _, e in v])
            out[str(k)] = dict(median=float(np.median(tot)), share_half_tear=float(np.mean(tot >= 0.5)),
                               median_max_joint=float(np.median(mx)), n=len(tot))
        S[f"{m}-traj"] = out
        print(f"{m} trajectory:", {k: (round(v["median"], 2), round(v["share_half_tear"], 2)) for k, v in out.items()})
    json.dump(S, open("results/summary_diffusion.json", "w"), indent=1, default=float)
