"""Figure 2: networks read an impossible figure one joint at a time.
(a) total excess tear sum_j e_j on the impossible image (set A) per network (samplers: all noise seeds), median and
    95% cluster-bootstrap interval over the 8 layout bins; the mean of the three objects and every single object
    have sum 1; making the ordinary real-joint step at every joint gives 0.
(b) joint locality (scripts/analyze_joints.py), in units of the network's own tear: the impossible image vs the
    matched possible control at the two shared joints (tick: the same against a control 45 deg away), and at the
    joint that differs, against the control's tear there and against that joint's ordinary step.
(c) gap morph: excess at the revealed joint vs the number of pixels the reveal changes (tau = 1 is the control
    itself, so e = 1 there by definition).
    Panel (a) also shows, hollow, the Marigold samplers with 4 and 10 DDIM steps (scripts/analyze_diffusion.py)."""
import os, sys, csv, json, numpy as np, matplotlib
from collections import defaultdict
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import MODELS, setup, short

setup()
ORDER = ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft", "mg11-1", "mg10-1"]
MODELS["mg11-1"] = dict(label="Marigold v1.1 (1 step)", color="#c51b7d")
MODELS["mg10-1"] = dict(label="Marigold v1.0 (1 step)", color="#f768a1")
MARK = {"dav2-s": "o", "dav2-b": "s", "dav2-l": "D", "dpt-l": "^", "zoe": "v", "lotus-d": "o", "lotus-g": "s",
        "e2eft": "D", "mg11-1": "^", "mg10-1": "v"}
LS = {m: "-" if m.startswith("dav2") else ("--" if m in ("dpt-l", "zoe") else ":") for m in ORDER}

if __name__ == "__main__":
    S = json.load(open("results/summary_excess.json"))
    L = json.load(open("results/summary_joints.json"))
    D = json.load(open("results/summary_diffusion.json")) if os.path.exists("results/summary_diffusion.json") else {}
    multi = [k for k in ("mg11-4", "mg11-10", "mg10-10") if k in D]
    ex = list(csv.DictReader(open("results/excess_rows.csv")))
    px = defaultdict(list)
    for r in csv.DictReader(open("results/morph_pixels.csv")):
        px[float(r["tau"])].append(int(r["changed_px"]))
    px = {t: np.median(v) for t, v in px.items()}
    models = [m for m in ORDER if m in S]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw=dict(width_ratios=[1.4, 1.2, 1.3]))
    # (a) ten networks at one step, then the Marigold samplers with more DDIM steps (hollow)
    ax = axes[0]
    for i, m in enumerate(models):
        v, lo, hi = S[m]["A"]["sum_e"]
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt=MARK[m], color=MODELS[m]["color"], ms=3.5, capsize=2, lw=1)
    confs = [(k, l) for k, l in [("mg11-4", "v1.1, 4 steps"), ("mg11-10", "v1.1, 10 steps"), ("mg10-10", "v1.0, 10 steps")] if k in D]
    x0 = len(models) + 0.6
    for i, (k, l) in enumerate(confs):
        v, lo, hi = D[k]["sum_e"]
        col = MODELS["mg11-1"]["color"] if k.startswith("mg11") else MODELS["mg10-1"]["color"]
        ax.errorbar(x0 + i, v, yerr=[[v - lo], [hi - v]], fmt="o", mfc="white", color=col, ms=3.5, capsize=2, lw=1)
    if confs:
        ax.axvline(len(models) - 0.2, color="0.75", lw=0.6)
    n = x0 + len(confs) if confs else len(models)
    ax.axhspan(0.97, 1.03, color="#e34a33", alpha=0.25, lw=0)
    ax.text(n - 0.5, 0.93, "mean or one object", ha="right", va="top", fontsize=5.8, color="#b2182b",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=3)
    ax.axhline(0, color="0.4", lw=0.7, ls=":")
    ax.text(n - 0.5, -0.58, "ordinary step at every joint: 0", ha="right", va="bottom", fontsize=5.8, color="0.3",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=3)
    ticks = list(range(len(models))) + [x0 + i for i in range(len(confs))]
    ax.set_xticks(ticks); ax.set_xticklabels([short(m) for m in models] + [l for _, l in confs], rotation=60, ha="right",
                                             fontsize=5.6, rotation_mode="anchor")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(-0.62, 1.15); ax.set_ylabel(r"total excess tear $\Sigma_j e_j$")
    ax.set_title("(a) impossible image: who pays?")
    # (b)
    ax = axes[1]
    x = np.arange(len(models)); w = 0.27
    bars = [("shared", "shared joints", "#4d4d4d"), ("vs_tear", "differing: vs tear", "#e34a33"),
            ("vs_real", "differing: vs ordinary", "#6baed6")]
    for j, (key, name, col) in enumerate(bars):
        ax.bar(x + (j - 1) * w, [L[m][key][0] for m in models], w, color=col, label=name)
    ax.plot(x - w, [L[m]["null_shared"][0] for m in models], "_", color="k", ms=5, mew=1.0,
            label="shared, other layout")
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=60, ha="right", fontsize=5.8, rotation_mode="anchor")
    ax.set_ylabel("|jump difference| (own tear)"); ax.set_title("(b) each joint is read locally")
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.02), handlelength=1.0,
              columnspacing=0.6, borderaxespad=0.1)
    ax.set_ylim(0, 1.6); ax.set_yticks(np.arange(0, 1.3, 0.25))
    ax.spines["left"].set_bounds(0, 1.25)      # the axis stops at the last tick, below the legend
    # (c)
    ax = axes[2]
    taus = sorted(px)
    for m in models:
        rows = [r for r in ex if r["model"] == m and r["set"] == "C"]
        if not rows:
            continue
        ys, lo, hi = [], [], []
        for t in taus:
            e = np.array([float(r[f"e{int(r['possible']) + 1}"]) for r in rows if abs(float(r["tau"]) - t) < 1e-6])
            ys.append(np.median(e))
            bs = [np.median(np.random.default_rng(i).choice(e, len(e))) for i in range(300)]
            lo.append(np.percentile(bs, 5)); hi.append(np.percentile(bs, 95))
        xs = [max(px[t], 10) for t in taus]
        ax.plot(xs, ys, marker=MARK[m], ls=LS[m], color=MODELS[m]["color"], ms=2.2, lw=1, label=short(m))
        ax.fill_between(xs, lo, hi, color=MODELS[m]["color"], alpha=0.08, lw=0)
    ax.set_xscale("log"); ax.set_xticks([10, 24, 72, 244, 1363, 3535])
    ax.set_xticklabels(["0", "24", "72", "244", "1.4k", "3.5k"], fontsize=6.3); ax.minorticks_off()
    ax.text(15.5, -0.36, "//", fontsize=6, ha="center", va="center", color="0.3")
    ax.axhline(1, color="#e34a33", lw=0.7, ls="--"); ax.axhline(0, color="0.4", lw=0.7, ls=":")
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("pixels of the gap revealed"); ax.set_ylabel("excess tear at revealed joint")
    ax.set_title("(c) revealing the gap restores the tear")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", bbox_to_anchor=(1.0, 1.02), handlelength=2.2)
    fig.tight_layout(w_pad=0.4)
    os.makedirs("figures/out", exist_ok=True)
    fig.savefig("figures/out/fig_main.pdf"); fig.savefig("figures/out/fig_main.png", dpi=220)
