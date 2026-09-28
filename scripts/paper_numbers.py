"""Every number in the report that no other script prints (run after analyze_excess.py).

- image counts per set, and the subset the samplers were run on;
- read-out accuracy on the ground-truth profiles (results/gt_profiles.csv);
- readability gate: share of possible controls whose most positive jump (largest jump back) is at the gap;
- ordinary joint steps: rho_j = -r_j / (g_j - r_j), the excess a continuous map (no jump at joint j) would score;
  sum_j rho_j is the total a continuous map would get;
- a belief mixing upright and depth-reversed objects, with the network's reversed reading taken as its upright
  reading negated, can reproduce the ordinary step at every joint only if sum_j rho_j <= 1/2 (then it needs
  P(rev) - P(up) = rho / (1 - rho));
- half-tear pixel counts of the gap morph (log-pixel interpolation of the median excess at the revealed joint);
- totals on the shifted/zoomed (D) and restyled (E) images, which have no matching controls;
- read-out sensitivity bound; Marigold's seed-averaged one-step totals; half-tear pixels with more DDIM steps;
- how often the bars follow the objects' slope on the controls and reverse it on the impossible image.
usage: python scripts/paper_numbers.py   (after analyze_excess.py and analyze_diffusion.py)"""
import os, sys, csv, json, numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load
from pdepth.loop import bars_and_joints

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_excess import RUNS, references, lay, P

ORDER = ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft", "mg11-1", "mg10-1"]

if __name__ == "__main__":
    S = json.load(open("results/summary_excess.json"))
    rows_db = load("results/readout_dav2-b.csv")
    print("images per set:", dict(Counter(r["set"] for r in rows_db)), " total", len(rows_db))
    smp = load("results/mg11_1step.csv")
    print("sampler runs (Marigold v1.1): images per set", dict(Counter(r["set"] for r in smp if r["seed"] == 0)),
          " seeds", len({r["seed"] for r in smp}))
    print("fraction of the image for 95 px: %.3f%%" % (100 * 95 / 512 ** 2))
    G = list(csv.DictReader(open("results/gt_profiles.csv")))
    T = 5 * np.sqrt(3)
    err = max(np.abs(bars_and_joints(np.array([float(r[f"prof{j}"]) for j in range(15)]))[1]
                     - T * np.eye(3)[int(r["k"])]).max() / T for r in G)
    print(f"read-out on the ground truth ({len(G)} profiles): largest joint error {100 * err:.1f}% of a tear")

    px = defaultdict(list)
    for r in csv.DictReader(open("results/morph_pixels.csv")):
        px[float(r["tau"])].append(int(r["changed_px"]))
    px = {t: float(np.median(v)) for t, v in px.items()}

    print(f"\n{'network':8s} gate  rho_j (median over layouts)   sum rho  mixture P(rev)-P(up)  half-tear px")
    for m in ORDER:
        if m not in RUNS or not os.path.exists(RUNS[m]):
            continue
        rows = [r for r in load(RUNS[m]) if r.get("seed", 0) != -1]
        B = [r for r in rows if r["set"] == "B"]
        gate = np.mean([np.argmax(bars_and_joints(np.array([r[p] for p in P], float))[1]) == int(r["possible"]) for r in B])
        ref = references(rows)
        rho = np.array([-real / (tear - real) for real, tear in ref.values()])
        rs = np.median(rho.sum(1))
        mix = f"{rs / (1 - rs):.2f}" if rs <= 0.5 else ("at bound" if rs < 0.51 else "impossible")
        c = S[m].get("C_ek", {})
        ts = sorted(float(t) for t in c)
        xs, ys = [np.log(max(px[t], 1)) for t in ts], [c[f"{t:.2f}"] for t in ts]
        half = next((np.exp(xs[i - 1] + (0.5 - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))
                     for i in range(1, len(ts)) if ys[i - 1] < 0.5 <= ys[i]), np.nan)
        print(f"{m:8s} {gate:4.2f}  {np.round(np.median(rho, 0), 2)}   {rs:5.2f}   {mix:>10s}   {half:7.0f}")

    print("\nshifted/zoomed (D) and restyled (E) images, total excess against the centred textured controls:")
    for m in ORDER:
        d = S.get(m, {})
        print(f"  {m:8s}", "  ".join(f"{st} {d[st]['sum_e'][0]:+.2f}" for st in ("D", "E") if st in d))
    sens = list(csv.DictReader(open("results/sensitivity.csv"))) if os.path.exists("results/sensitivity.csv") else []
    if sens:
        print("\nread-out sensitivity (largest change of the median total vs erosion 3 px, median), networks re-read:")
        for m in sorted({r["model"] for r in sens}):
            R = [r for r in sens if r["model"] == m]
            base = [float(r["sum_e_median"]) for r in R if r["erode"] == "3" and r["stat"] == "median"][0]
            print(f"  {m:8s} {max(abs(float(r['sum_e_median']) - base) for r in R):.3f}")

    from analyze_excess import excess
    print("\nMarigold one-step, seed-averaged output (posterior-mean estimate) vs seed-mean controls, set A:")
    for t in ("mg11", "mg10"):
        sm = load(f"results/{t}_1step_seedmean.csv")
        print(f"  {t}: total excess {np.median([o['e'].sum() for o in excess(sm) if o['r']['set'] == 'A']):.2f}")

    D = json.load(open("results/summary_diffusion.json"))
    print("\nhalf-tear pixels of the gap morph with more DDIM steps:")
    for k in ("mg11-4-morph", "mg11-10-morph", "mg10-10-morph"):
        c = D.get(k)
        if not c:
            continue
        ts = sorted(float(t) for t in c)
        xs, ys = [np.log(max(px[t], 1)) for t in ts], [c[f"{t:.2f}"] for t in ts]
        half = next((np.exp(xs[i - 1] + (0.5 - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))
                     for i in range(1, len(ts)) if ys[i - 1] < 0.5 <= ys[i]), np.nan)
        print(f"  {k[:-6]:8s} {half:.0f}")

    print("\nbar slopes: share of controls whose mean bar slope follows the objects' (< 0), and of impossible"
          " outputs whose bars reverse it (> 0):")
    for m in ORDER:
        if m not in RUNS or not os.path.exists(RUNS[m]):
            continue
        rows = [r for r in load(RUNS[m]) if r.get("seed", 0) != -1]
        b = lambda st: np.array([bars_and_joints(np.array([r[p] for p in P], float))[0].mean() for r in rows if r["set"] == st])
        print(f"  {m:8s} controls {np.mean(b('B') < 0):.2f}   impossible reversed {np.mean(b('A') > 0):.2f}")
