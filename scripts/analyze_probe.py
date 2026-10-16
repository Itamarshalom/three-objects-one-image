"""Erasure probe (scripts/run_probe.py): where does a network put the tear when one junction is erased?

Excess tears e_j are computed as in scripts/analyze_excess.py, against the network's own intact possible controls;
an erased junction's jump is the next bar's line minus the previous bar's line, both fitted on their visible cubes.
Reported per network:
  impossible, intact    : total excess and the share of images whose bars run against the objects' (mean slope > 0)
  impossible, j erased  : e at the erased junction, |e| at the two visible ones, the share where the largest jump
                          sits at the erased junction, and the share whose bars run with the objects' (mean slope < 0)
  control k, j erased   : e at the visible gap k (j != k) and at the erased junction
  recovery              : with junction j erased, the share of images whose visible loop profile correlates best with
                          object j (among the three objects and their mean); intact images: the best correlation
  impossibility score   : S = mean over j of (e_j with j erased - e_j intact); impossible vs controls (AUC); and the
                          scale-free share of the three erasures that move the largest jump to the erased junction
                          (1 if each erasure releases a tear there; 1/3 for a control, whose gap keeps it)
usage: python scripts/analyze_probe.py   -> results/summary_probe.json"""
import os, sys, csv, glob, json, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load, boot_ci
from pdepth.loop import bars_and_joints

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_excess import references, lay, P

ORDER = ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft", "mg11-1", "mg10-1"]


def gt_profiles(path="results/gt_profiles.csv"):
    G = defaultdict(lambda: np.full((3, 15), np.nan))
    for r in csv.DictReader(open(path)):
        G[r["stim"]][int(r["k"])] = [float(r[p]) for p in P]
    return G


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() >= 4 else np.nan


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def summarise(rows, G):
    ref = references([r for r in rows if r["cond"] == "none"])
    by = {}
    for r in rows:
        if lay(r) not in ref:
            continue
        real, tear = ref[lay(r)]
        prof = np.array([r[p] for p in P], float)
        b, J = bars_and_joints(prof)
        g = G.get(r["stim"])
        cs = np.array([corr(prof, g[k]) for k in range(3)] + [corr(prof, g.mean(0))]) if g is not None else np.full(4, np.nan)
        by[(r["stim"], r["seed"], r["cond"])] = dict(r=r, e=(J - real) / (tear - real), slope=np.nanmean(b), corr=cs)
    out, grp = {}, lambda o: o["r"]["angle"] - o["r"]["angle"] % 15
    A0 = [o for (s, sd, c), o in by.items() if o["r"]["set"] == "A" and c == "none"]
    if not A0:
        return None
    tot = np.array([o["e"].sum() for o in A0])
    out["impossible_intact"] = dict(sum_e=boot_ci(tot, np.median, groups=[grp(o) for o in A0]),
                                    reversed=float(np.mean([o["slope"] > 0 for o in A0])),
                                    best_corr=float(np.median([np.nanmax(o["corr"][:3]) for o in A0])),
                                    n=len(A0))
    rel, vis, top, fwd, rec, S, M = [], [], [], [], [], defaultdict(list), defaultdict(list)
    for (s, sd, c), o in by.items():
        if c == "none" or o["r"]["set"] not in "AB":
            continue
        j = int(c[1])
        base = by.get((s, sd, "none"))
        if base is not None:
            S[(o["r"]["set"], s, sd)].append(o["e"][j] - base["e"][j])
        M[(o["r"]["set"], s, sd)].append(np.nanargmax(o["e"]) == j)
        if o["r"]["set"] == "A":
            rel.append((o["e"][j], grp(o))); vis.extend(np.abs(np.delete(o["e"], j)))
            top.append(np.nanargmax(o["e"]) == j); fwd.append(o["slope"] < 0)
            rec.append(np.nanargmax(o["corr"]) == j)
    e_rel = np.array([x for x, _ in rel])
    out["impossible_erased"] = dict(e_erased=boot_ci(e_rel, np.median, groups=[g for _, g in rel]),
                                    abs_e_visible=float(np.median(vis)), top_at_erased=float(np.mean(top)),
                                    bars_with_objects=float(np.mean(fwd)), recovered_object=float(np.mean(rec)),
                                    n=len(rel))
    gap, oth, own, keep = [], [], [], []
    for (s, sd, c), o in by.items():
        if c == "none" or o["r"]["set"] != "B":
            continue
        j, k = int(c[1]), int(o["r"]["possible"])
        if j == k:
            own.append(o["e"][k])
        else:
            gap.append(o["e"][k]); oth.append(o["e"][j]); keep.append(np.nanargmax(o["e"]) == k)
    out["control_erased"] = dict(e_gap_other_erased=float(np.median(gap)), e_erased_ordinary=float(np.median(oth)),
                                 e_gap_erased=float(np.median(own)), top_stays_at_gap=float(np.mean(keep)),
                                 n=len(gap) + len(own))
    sA = [np.mean(v) for (st, s, sd), v in S.items() if st == "A" and len(v) == 3]
    sB = [np.mean(v) for (st, s, sd), v in S.items() if st == "B" and len(v) == 3]
    mA = [np.mean(v) for (st, s, sd), v in M.items() if st == "A" and len(v) == 3]
    mB = [np.mean(v) for (st, s, sd), v in M.items() if st == "B" and len(v) == 3]
    out["score"] = dict(impossible=float(np.median(sA)), control=float(np.median(sB)), auc=auc(sA, sB),
                        moved_impossible=float(np.mean(mA)), moved_control=float(np.mean(mB)),
                        n_impossible=len(sA), n_control=len(sB))
    return out


if __name__ == "__main__":
    G = gt_profiles()
    S = {}
    files = {os.path.basename(f)[6:-4]: f for f in glob.glob("results/probe_*.csv")}
    for m in [m for m in ORDER if m in files] + sorted(set(files) - set(ORDER)):
        rows = [r for r in load(files[m]) if r.get("seed", 0) != -1]
        res = summarise(rows, G)
        if res is None:
            continue
        S[m] = res
        a, e, c, sc = res["impossible_intact"], res["impossible_erased"], res["control_erased"], res["score"]
        print(f"{m:8s} intact: sum e {a['sum_e'][0]:+.2f}, bars reversed {a['reversed']:.2f} | erased j: e_j "
              f"{e['e_erased'][0]:+.2f} [{e['e_erased'][1]:+.2f},{e['e_erased'][2]:+.2f}], |e| visible {e['abs_e_visible']:.2f}, "
              f"largest jump at j {e['top_at_erased']:.2f}, bars with objects {e['bars_with_objects']:.2f}, object j "
              f"recovered {e['recovered_object']:.2f} | control: gap {c['e_gap_other_erased']:+.2f} / erased ordinary "
              f"{c['e_erased_ordinary']:+.2f} / gap erased {c['e_gap_erased']:+.2f}, largest stays at gap "
              f"{c['top_stays_at_gap']:.2f} | score {sc['impossible']:+.2f} vs {sc['control']:+.2f}, AUC {sc['auc']:.2f};"
              f" erasures moving the largest jump {sc['moved_impossible']:.2f} vs {sc['moved_control']:.2f}")
    json.dump(S, open("results/summary_probe.json", "w"), indent=1)
