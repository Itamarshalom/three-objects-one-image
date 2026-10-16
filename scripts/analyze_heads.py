"""Matched read-out heads (scripts/train_heads.py, run by scripts/run_heads.py): does a head that outputs depth
changes (1-form) keep the impossible figure that a head outputting depth (0-form) bends?

For every head output in outputs/heads_out/<name>/ (the impossible images A, the possible controls B and the closed
possible rings), the 1-form read-out of pdepth/oneform.py on the figure minus the pixels the head itself marks as
occlusion edges (edge probability >= 0.5): whether the loop is closed, the along-bar slopes (as a ratio to the exact
-1/sqrt(2)), the circulation Gamma/T (from the bars and around a pixel path), and, for closed loops, the cut at each
junction k compared with object k's exact loop profile. For a 0-form head g is the gradient of its depth, so its
circulation is ~0 by construction: the comparison is the point. Impossibility score: closed x |Gamma_bars| / T;
AUC impossible vs rings and vs controls. The 0-form heads' erasure probe goes through scripts/analyze_probe.py and
scripts/analyze_loops.py (key head-<name>).
The 0-form heads read as depth networks (depth_head_excess, from results/readout_head-0form-*.csv only, so it also
runs without outputs/heads_out): the excess tears of scripts/analyze_excess.py against the head's own controls, the
share of controls whose largest jump J is at their gap (as "gap found" in scripts/paper_numbers.py), and on the
impossible images the median largest e_j and sum of e_j (intervals resample 15-degree layout bins). Stored as "excess"
under the head's entry. The JSON is merged: heads not named on this run keep their entries, and so does a head with
fewer outputs here than its stored entry counts (the local copy holds only the figure's few outputs).
usage: python scripts/analyze_heads.py [head name ...]   -> results/summary_heads.json
       (no names: only the 0-form part, from the CSVs)"""
import os, sys, glob, csv, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdepth.oneform as of
from pdepth.readout import cube_masks
from pdepth.analysis import boot_ci, load
from pdepth.loop import bars_and_joints
from analyze_excess import excess

T = 5 * np.sqrt(3)
EXACT = -1 / np.sqrt(2)
P = [f"prof{j}" for j in range(15)]
OUT = "results/summary_heads.json"


def gt_profiles(path="results/gt_profiles.csv"):
    G = {}
    for r in csv.DictReader(open(path)):
        G.setdefault(r["stim"], np.full((3, 15), np.nan))[int(r["k"])] = [float(r[p]) for p in P]
    return G


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])) if len(pos) and len(neg) else np.nan


def stim_path(name):
    for d in ("data/stim2", "data/loops"):
        f = os.path.join(d, name + ".npz")
        if os.path.exists(f):
            return f


def one(name, out, gt):
    st = dict(np.load(stim_path(name)))
    st["set"] = str(st["set"])
    d = np.load(out)
    px, nb = of.pixel_size(st), int(st.get("n_bars", 3))
    mask = st["fg"] & (d["edge"] < 0.5)
    o = of.readout(d["g"].astype(float), mask, st["cube"], px, nb, masks=cube_masks(st["cube"], 5 * nb))
    r = dict(stim=name, fam=str(st["set"]), closed=bool(o["closed"]),
             slope=float(np.nanmean(o["slopes"]) / (EXACT if st["set"] in "AB" else 1)),
             gamma_bars=float(o["gamma_bars"] / T), gamma_path=float(o["gamma_path"] / T),
             edge_at_junctions=float(np.mean([d["edge"][np.isin(st["cube"], [5 * k - 1 if k else 5 * nb - 1, 5 * k])].mean()
                                              for k in range(nb)])))
    if st["set"] == "A" and o["closed"] and name in gt:
        best = []
        for c in o["cuts"]:
            g = gt[name]
            cs = [np.corrcoef(c["profile"], g[k])[0, 1] for k in range(3)]
            best.append(int(np.nanargmax(cs)) == c["k"])
        r["cut_gives_object"] = float(np.mean(best))
        r["gamma_cut"] = float(o["gamma_cut"] / T)
    return r


def depth_head_excess(S):
    """the 0-form heads' depth read out on the loop (results/readout_head-<name>.csv), scored like a depth network"""
    for fn in sorted(glob.glob("results/readout_head-0form-*.csv")):
        name = os.path.basename(fn)[len("readout_head-"):-4]
        rows = [r for r in load(fn) if r.get("seed", 0) != -1]
        B = [r for r in rows if r["set"] == "B"]
        gap = np.mean([int(np.argmax(bars_and_joints(np.array([r[p] for p in P], float))[1])) == int(r["possible"])
                       for r in B])
        A = [o for o in excess(rows) if o["r"]["set"] == "A"]
        grp = [o["r"]["angle"] - o["r"]["angle"] % 15 for o in A]
        res = dict(gap_found=float(gap), n_controls=len(B), n_impossible=len(A),
                   max_e=boot_ci([o["e"].max() for o in A], np.median, groups=grp),
                   sum_e=boot_ci([o["e"].sum() for o in A], np.median, groups=grp))
        S.setdefault(name, {})["excess"] = res
        m, s = res["max_e"], res["sum_e"]
        print(f"{name:16s} (0form, as a depth network) controls: largest jump at the gap {gap:.2f} (n={len(B)}) | "
              f"impossible: largest e {m[0]:+.2f} [{m[1]:+.2f},{m[2]:+.2f}], sum e {s[0]:+.2f} [{s[1]:+.2f},"
              f"{s[2]:+.2f}] (n={len(A)})")


if __name__ == "__main__":
    gt = gt_profiles() if sys.argv[1:] else None
    S = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in sys.argv[1:]:
        files = sorted(glob.glob(f"outputs/heads_out/{name}/*.npz"))
        files = [f for f in files if os.path.basename(f)[0] in "ABHR"]
        if not files:
            print(f"{name}: no outputs"); continue
        n_old = sum(S.get(name, {}).get("n", {}).values())
        if len(files) < n_old:                               # e.g. the few figure outputs copied off the VM
            print(f"{name}: only {len(files)} outputs here, stored entry has {n_old}: kept"); continue
        rows = [one(os.path.basename(f)[:-4], f, gt) for f in files]
        kind = str(np.load(files[0])["kind"])
        fam = lambda F: [r for r in rows if r["fam"] in F]
        A, B, R = fam("A"), fam("B"), fam("HR")
        score = lambda rs: [float(r["closed"]) * abs(r["gamma_bars"]) for r in rs]
        grp = [int(r["stim"].split("_a")[1][:3]) // 15 for r in A]
        res = dict(kind=kind, n=dict(A=len(A), B=len(B), rings=len(R)),
                   closed=dict(A=np.mean([r["closed"] for r in A]), B=np.mean([r["closed"] for r in B]),
                               rings=np.mean([r["closed"] for r in R])),
                   slope_ratio=dict(A=boot_ci([r["slope"] for r in A], np.median, groups=grp),
                                    B=float(np.median([r["slope"] for r in B]))),
                   gamma_bars=dict(A=boot_ci([r["gamma_bars"] for r in A], np.median, groups=grp),
                                   rings=float(np.median([r["gamma_bars"] for r in R]))),
                   gamma_path=dict(A=float(np.nanmedian([r["gamma_path"] for r in A])),
                                   rings=float(np.nanmedian([r["gamma_path"] for r in R]))),
                   edge_at_junctions=dict(A=float(np.median([r["edge_at_junctions"] for r in A])),
                                          B=float(np.median([r["edge_at_junctions"] for r in B]))),
                   cut_gives_object=float(np.nanmean([r.get("cut_gives_object", np.nan) for r in A])),
                   auc_vs_rings=auc(score(A), score(R)), auc_vs_controls=auc(score(A), score(B)))
        S[name] = dict(res, **{k: v for k, v in S.get(name, {}).items() if k == "excess"})   # kept if no CSV here
        c, sl, g = res["closed"], res["slope_ratio"], res["gamma_bars"]
        print(f"{name:16s} ({kind}) closed A {c['A']:.2f} B {c['B']:.2f} rings {c['rings']:.2f} | bar slope / exact A "
              f"{sl['A'][0]:+.2f} [{sl['A'][1]:+.2f},{sl['A'][2]:+.2f}] B {sl['B']:+.2f} | Gamma/T A {g['A'][0]:+.2f} "
              f"[{g['A'][1]:+.2f},{g['A'][2]:+.2f}] (path {res['gamma_path']['A']:+.2f}) rings {g['rings']:+.2f} | cut k -> "
              f"object k {res['cut_gives_object']:.2f} | AUC vs rings {res['auc_vs_rings']:.2f} vs controls "
              f"{res['auc_vs_controls']:.2f}")
    depth_head_excess(S)
    json.dump(S, open(OUT, "w"), indent=1, default=float)
