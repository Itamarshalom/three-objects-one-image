"""Normal networks vs depth networks with the same backbone: do the bars keep their orientation on the impossible image?

Normal networks (scripts/run_normals.py): one axis-sign convention per network, the one that best matches the exact
normals on the possible controls. Along-bar slope = change of depth per unit length along the bar (exact -1/sqrt(2)).
Depth networks (results/readout_*.csv, results/mg11_1step.csv): bar slope per cube step from the loop read-out.
For both, "relative tilt" = mean bar slope on an impossible image / mean bar slope on the possible controls of the same
15-degree layout bin and chirality: +1 = the bars run as on the controls, < 0 = they run the other way.
usage: python scripts/analyze_normals.py   -> results/summary_normals.json"""
import os, sys, csv, json, itertools, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load, boot_ci
from pdepth.loop import bars_and_joints

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_excess import lay, P

PAIRS = [("mg11-n", "results/mg11_1step.csv", "Marigold v1.1"), ("e2eft-n", "results/readout_e2eft.csv", "E2E-FT"),
         ("lotus-d-n", "results/readout_lotus-d.csv", "Lotus-D")]
TAGS = ["".join("+" if v > 0 else "-" for v in s) for s in itertools.product((1, -1), repeat=3)]
TRUE = -1 / np.sqrt(2)


def relative_tilt(items):
    """items: (row, mean slope); returns per impossible image the slope relative to its layout's controls"""
    ctl = defaultdict(list)
    for r, s in items:
        if r["set"] == "B":
            ctl[lay(r)].append(s)
    out = []
    for r, s in items:
        if r["set"] == "A" and lay(r) in ctl:
            out.append((s / np.mean(ctl[lay(r)]), r["angle"] - r["angle"] % 15))
    return out


if __name__ == "__main__":
    S = {}
    for nk, dfile, lab in PAIRS:
        fn = f"results/normals_{nk}.csv"
        if not os.path.exists(fn):
            continue
        R = load(fn)
        tag = max(TAGS, key=lambda t: np.mean([r["cos" + t] for r in R if r["set"] == "B"]))
        sl = [(r, np.mean([r[f"slope{j}{tag}"] for j in range(3)])) for r in R]
        rel = relative_tilt(sl)
        res = dict(convention=tag,
                   cos_control=float(np.median([r["cos" + tag] for r in R if r["set"] == "B"])),
                   cos_impossible=float(np.median([r["cos" + tag] for r in R if r["set"] == "A"])),
                   slope_control=float(np.median([s for r, s in sl if r["set"] == "B"])),
                   slope_impossible=float(np.median([s for r, s in sl if r["set"] == "A"])),
                   gain_impossible=float(np.median([s for r, s in sl if r["set"] == "A"]) / TRUE),
                   share_with_objects_impossible=float(np.mean([s < 0 for r, s in sl if r["set"] == "A"])),
                   rel_tilt=boot_ci([x for x, _ in rel], np.median, groups=[g for _, g in rel]))
        if os.path.exists(dfile):
            D = [r for r in load(dfile) if r.get("seed", 0) != -1]
            dsl = [(r, np.mean(bars_and_joints(np.array([r[p] for p in P], float))[0])) for r in D if r["set"] in "AB"]
            drel = relative_tilt(dsl)
            res["depth"] = dict(rel_tilt=boot_ci([x for x, _ in drel], np.median, groups=[g for _, g in drel]),
                                share_with_objects_impossible=float(np.mean([s < 0 for r, s in dsl if r["set"] == "A"])))
        S[nk] = res
        d = res.get("depth", {})
        print(f"{lab:13s} normals ({tag}): cos {res['cos_control']:.2f} controls / {res['cos_impossible']:.2f} impossible;"
              f" along-bar slope {res['slope_control']:+.3f} / {res['slope_impossible']:+.3f} (exact {TRUE:+.3f});"
              f" bars with the objects on impossible {res['share_with_objects_impossible']:.2f}; relative tilt "
              f"{res['rel_tilt'][0]:+.2f} [{res['rel_tilt'][1]:+.2f},{res['rel_tilt'][2]:+.2f}]"
              + (f" | depth twin: relative tilt {d['rel_tilt'][0]:+.2f} [{d['rel_tilt'][1]:+.2f},{d['rel_tilt'][2]:+.2f}],"
                 f" bars with the objects {d['share_with_objects_impossible']:.2f}" if d else ""))
    json.dump(S, open("results/summary_normals.json", "w"), indent=1)
