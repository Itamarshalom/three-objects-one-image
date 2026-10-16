"""Summary of the attention cut inside Depth Anything V2.

Attention cut (scripts/run_attention_cut.py -> results/attn*_<model>.csv): the statistics of the pixel erasure probe
(scripts/analyze_probe.py, excess tears against the network's own intact controls) with "junction j erased" replaced
by "junction j cut out of attention": e at the cut junction, |e| at the two others, the share of images whose largest
jump sits at the cut junction, the share whose bars run with the objects (mean slope < 0), the share whose loop profile
best matches object j, and the bar slope ratio rho = mean bar slope / the intact controls' (+1 = the objects', < 0 =
reversed); for the possible controls, e at their gap and at a cut REAL joint (what the cut does to a loop that is not
impossible: the part of e_j that is not specific to the impossible figure). Read twice: on the full cube masks (the
end cubes of the two bars at j are drawn from the isolated tokens and count) and with the pixel probe's hole removed
(the cubes the pixel erasure leaves visible); the pixel erasure of the same images (results/probe_<model>.csv) is
printed with them. Tables of the closed rings (n_bars != 3) are skipped.
usage: python scripts/analyze_attention.py [csv files...]   (default results/attn*_*.csv)
-> results/summary_attention.json
"""
import os, sys, glob, json, numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdepth.analysis import load
from pdepth.loop import bars_and_joints
from analyze_excess import lay, P
from analyze_probe import gt_profiles, summarise

H = [f"hprof{j}" for j in range(15)]
prof = lambda r, cols=P: np.array([r[p] for p in cols], float)


def control_slopes(rows):
    """per layout: the intact possible controls' mean bar slope (all bars, all three controls)"""
    s = defaultdict(list)
    for r in rows:
        if r["set"] == "B" and r["cond"] in ("none", "ref"):
            s[lay(r)].append(np.nanmean(bars_and_joints(prof(r))[0]))
    return {k: np.mean(v) for k, v in s.items() if len(v) >= 3}


def rho_cut(rows):
    """median rho of the impossible images, intact and with a junction cut"""
    ref, out = control_slopes(rows), defaultdict(list)
    for r in rows:
        if r["set"] == "A" and lay(r) in ref:
            rho = np.nanmean(bars_and_joints(prof(r))[0]) / ref[lay(r)]
            out["intact" if r["cond"] == "none" else "cut"].append(rho)
    return {k: float(np.median(v)) for k, v in out.items()}


def show(label, res, rho):
    if res is None:
        print(f"  {label:26s}: no layout with all three intact controls"); return
    e, c, sc = res["impossible_erased"], res["control_erased"], res["score"]
    print(f"  {label:26s}: e_j {e['e_erased'][0]:+.2f} [{e['e_erased'][1]:+.2f},{e['e_erased'][2]:+.2f}] | |e| others "
          f"{e['abs_e_visible']:.2f} | largest at j {e['top_at_erased']:.2f} | bars with objects "
          f"{e['bars_with_objects']:.2f}, rho {rho.get('cut', np.nan):+.2f} (intact {rho.get('intact', np.nan):+.2f}) | "
          f"object j {e['recovered_object']:.2f} | n={e['n']}")
    print(f"  {'':26s}  controls: e at the gap {c['e_gap_other_erased']:+.2f} (gap itself cut "
          f"{c['e_gap_erased']:+.2f}), at a cut real joint {c['e_erased_ordinary']:+.2f}, largest stays at the gap "
          f"{c['top_stays_at_gap']:.2f} | score {sc['impossible']:+.2f} vs {sc['control']:+.2f}, AUC {sc['auc']:.2f}")


def attention(rows, model, G):
    stims = {r["stim"] for r in rows}
    hole = [dict(r, **{p: r[h] for p, h in zip(P, H)}) for r in rows]
    out = {"full": summarise(rows, G), "hole": summarise(hole, G), "rho_full": rho_cut(rows), "rho_hole": rho_cut(hole),
           "n_tok": float(np.median([r["n_tok"] for r in rows if r["cond"] != "none"]))}
    show("cut, full masks", out["full"], out["rho_full"])
    show("cut, probe's hole removed", out["hole"], out["rho_hole"])
    pf = os.path.join(ROOT, f"results/probe_{model}.csv")
    if os.path.exists(pf):
        pix = [r for r in load(pf) if r["stim"] in stims and r.get("seed", 0) == 0]
        out["pixel"], out["rho_pixel"] = summarise(pix, G), rho_cut(pix)
        show("pixel erasure, same images", out["pixel"], out["rho_pixel"])
    return out


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("results/attn*_*.csv"))
    G, S = gt_profiles(os.path.join(ROOT, "results/gt_profiles.csv")), {}
    for f in files:
        label, model = os.path.basename(f)[:-4].rsplit("_", 1)
        rows = load(f)
        if label.startswith("attn") and any(r.get("n_bars", 3) != 3 for r in rows):
            print(f"{model} {label}: closed rings (n_bars != 3), skipped: this summary is written for the Penrose loop")
        elif label.startswith("attn"):
            n_tok = np.median([r["n_tok"] for r in rows if r["cond"] != "none"])
            print(f"{model} attention cut ({label}, cover {rows[-1]['cover']}, median {n_tok:.0f} tokens): e at the cut "
                  f"junction j [95% CI] | |e| at the other two | ...")
            S.setdefault(model, {})[label] = attention(rows, model, G)
    json.dump(S, open("results/summary_attention.json", "w"), indent=1, default=float)
