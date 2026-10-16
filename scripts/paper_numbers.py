"""Every number in the report that no other script prints, from results/ only (CPU, a few seconds).

The other numbers are printed by scripts/analyze_excess.py (total excess tear), analyze_probe.py and analyze_loops.py
(erasure, closed rings), analyze_normals.py, analyze_glue.py, analyze_attention.py, analyze_heads.py, and by
figures/fig_main.py and figures/fig_hero.py, which print every value they plot. This script prints, per network where
the report gives a range over networks (Sec. 2.2: "the repository lists each"):
- counts: images per set (288 impossible images, 48 controls), the samplers' subset (16 impossible images, several
  noise seeds), the closed rings (64), the training scenes (1500: probes fitted on 1200 and checked on the other 300,
  heads trained on 1400 and validated on 100);
- the networks bend instead of cutting: the share of impossible outputs whose mean bar slope is reversed (> 0) and of
  controls whose mean bar slope has the objects' sign (< 0); the share of controls whose largest jump is at their gap;
  the pixels in which a control differs from the impossible image (results/morph_pixels.csv at the full trim, tau 1);
- layer probes (results/summary_layers_<dav2-s|dav2-b>.json): the depth-change probe's slope gain k before L6 (below
  0.1 no bar slope is found, and no calibrated value is given), the depth probes' slope on the controls at L6-L7 (why
  Fig. 2c draws them from L8), the largest slope on the controls of any probe before the first layer Fig. 2c draws
  (depth-change probe L0-L5, depth probe L0-L7; the caption's "under 0.15", raw), the depth-change probe's rho and
  loop integral Gamma from L7 to F1 (impossible images, controls, rings), the depth probe's rho at L8-L12, in the
  decoder and on the network's own output, and on the bar pixels that are identical to a control's: the change of the
  tokens (L7-L12) and both read-outs (L8-L12);
- gluing (results/summary_glue.json, the eight networks of the report): the best junction weight and its margin over
  weight 1;
- matched heads (results/summary_heads.json, two seeds each): the depth-change head's Gamma, rings, cuts, the sum over
  one lap (median image, Fig. 1), closure and AUC on the controls; both heads' bar slopes and their ratio, impossible
  images over controls (the depth head against its own controls); the depth head's excess tears.
usage: python scripts/paper_numbers.py"""
import os, sys, re, csv, json, numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pdepth.analysis import load
from pdepth.loop import bars_and_joints

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_excess import RUNS, P

ORDER = ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft", "mg11-1", "mg10-1"]
GOOD = ["dav2-s", "dav2-b", "dav2-l", "lotus-d"]          # controls read with the objects' sign (Sec. 3, first paragraph)
LAYERS = [f"L{i}" for i in range(13)] + ["F1", "F2", "F3", "F4", "OUT"]


def span(v, fm=".2f"):
    """the range of the values, as the report writes it"""
    v = [x for x in v if x is not None]
    return f"{min(v):{fm}} to {max(v):{fm}}" if v else "-"


def between(a, b):
    """the layers from a to b (inclusive)"""
    return LAYERS[LAYERS.index(a):LAYERS.index(b) + 1]


if __name__ == "__main__":
    os.chdir(ROOT)                                        # analyze_excess.RUNS are relative to the repo root

    # ---------------------------------------------------------------- counts
    rows_db = load("results/readout_dav2-b.csv")
    print("images per set (A impossible, B possible controls):", dict(Counter(r["set"] for r in rows_db)))
    print("samplers (one denoising step):")
    for m in ("lotus-g", "mg11-1", "mg10-1"):
        rows = [r for r in load(RUNS[m]) if r.get("seed", 0) != -1]
        pr = list(csv.DictReader(open(f"results/probe_{m}.csv")))
        print(f"  {m:7s} read-out: {len({r['stim'] for r in rows if r['set'] == 'A'})} impossible images,"
              f" {len({r['seed'] for r in rows})} seeds | erasure probe: "
              f"{len({r['stim'] for r in pr if r['set'] == 'A'})} impossible images, {len({r['seed'] for r in pr})} seeds")
    H = json.load(open("results/summary_heads.json"))
    lay = {n: json.load(open(f"results/summary_layers_{n}.json")) for n in ("dav2-s", "dav2-b")}
    ex = lay["dav2-s"]["exact"]
    print(f"closed possible rings: {H['1form-s-seed0']['n']['rings']} (hexagons {ex['H']['n']}, rhombi {ex['R']['n']})")
    for n, L in lay.items():
        g = L["scene_groups"]
        test = g[max(g, key=int)]                         # the last group: scenes with index % 5 == 0
        print(f"layer probes {n}: {L['n_scenes']} training scenes, fitted on {L['n_scenes'] - test}"
              f" ({len(g) - 1}-fold CV), checked on {test}")
    n_sc = int(re.search(r"make_scenes\.py data/scenes (\d+)", open("scripts/queue_vm_heads.sh").read()).group(1))
    n_val = int(re.search(r'"--n-val", type=int, default=(\d+)', open("scripts/train_heads.py").read()).group(1))
    print(f"heads: {n_sc} scenes (scripts/queue_vm_heads.sh), trained on {n_sc - n_val}, validated on {n_val}"
          f" (scripts/train_heads.py --n-val)")

    # ---------------------------------------------------------------- the networks bend instead of cutting
    print("\nbar slopes and gaps: share of impossible outputs whose mean bar slope is reversed (> 0), of controls whose"
          " mean bar slope has the objects' sign (< 0), and of controls whose largest jump is at their gap:")
    rev, ctl, gate = {}, {}, {}
    for m in ORDER:
        rows = [r for r in load(RUNS[m]) if r.get("seed", 0) != -1]
        bj = lambda r: bars_and_joints(np.array([r[p] for p in P], float))
        rev[m] = np.mean([bj(r)[0].mean() > 0 for r in rows if r["set"] == "A"])
        ctl[m] = np.mean([bj(r)[0].mean() < 0 for r in rows if r["set"] == "B"])
        gate[m] = np.mean([np.argmax(bj(r)[1]) == int(r["possible"]) for r in rows if r["set"] == "B"])
        print(f"  {m:8s} impossible reversed {rev[m]:.2f} | controls with the objects {ctl[m]:.2f} | gap found {gate[m]:.2f}")
    print(f"  ranges: reversed {span(rev.values())}; controls {span(ctl[m] for m in GOOD)} (DA-S/B/L, Lotus-D),"
          f" {span(ctl[m] for m in ORDER if m not in GOOD)} (the other six); gap found {span(gate.values())}")
    px = [int(r["changed_px"]) for r in csv.DictReader(open("results/morph_pixels.csv")) if float(r["tau"]) == 1.0]
    print(f"pixels in which a control differs from its impossible image: median {np.median(px):.0f}"
          f" ({min(px)}-{max(px)}, {len(px)} controls)")

    # ---------------------------------------------------------------- layer probes
    print("\nlayer probes (calibrated; Gamma in units of T, exact 1 on impossible images and controls, 0 on rings):")
    R = {}
    for n, L in lay.items():
        Y = L["layers"]
        one = lambda l: Y[l]["oneform"]
        zero = lambda l: Y[l]["zeroform"]
        ip = lambda l, p: Y[l]["identical_pixels"][p]["A_over_B"]
        R[n] = dict(
            k_early=[one(l)["gain"] for l in between("L0", "L5")],
            ctl0=[zero(l)["B"]["ratio"] for l in ("L6", "L7")],
            ctl0_cal=[zero(l)["B"].get("ratio_cal") for l in ("L6", "L7")],
            rho1=[one(l)["A"]["rho"] for l in between("L7", "F1")],
            gA=[-one(l)["A"]["gamma"] for l in between("L7", "F1")],
            gB=[-one(l)["B"]["gamma"] for l in between("L7", "F1")],
            gR=[abs(one(l)["rings"]["gamma"]) for l in between("L7", "F1")],
            rho0=[zero(l)["A"]["rho"] for l in between("L8", "L12")],
            rho0_dec=[zero(l)["A"]["rho"] for l in between("F1", "F4")],
            rho0_out=zero("OUT")["A"]["rho"],
            rho1_F4=one("F4")["A"]["rho"],
            tok=[Y[l]["identical_pixels"]["token_change"] for l in between("L7", "L12")],
            ip1=[ip(l, "oneform") for l in between("L8", "L12")],
            ip0=[ip(l, "zeroform") for l in between("L8", "L12")])
        r = R[n]
        print(f"  {n}: depth-change probe slope gain k at L0-L5 {span(r['k_early'], '.3f')} (k < 0.1: no bar slope)\n"
              f"    depth probe on the controls at L6, L7: slope / exact {r['ctl0'][0]:+.3f}, {r['ctl0'][1]:+.3f} (raw),"
              f" {r['ctl0_cal'][0]:+.3f}, {r['ctl0_cal'][1]:+.3f} (calibrated)\n"
              f"    depth-change probe L7-F1: rho {span(r['rho1'])}, Gamma {span(r['gA'])} (controls {span(r['gB'])}),"
              f" rings |Gamma| <= {max(r['gR']):.2f}; at F4 rho {r['rho1_F4']:.2f}\n"
              f"    depth probe: rho {span(r['rho0'])} (L8-L12), {span(r['rho0_dec'])} (F1-F4), network output"
              f" {r['rho0_out']:+.2f}\n"
              f"    identical bar pixels: tokens differ by {span(r['tok'])} of their spread (L7-L12); read-out / control's"
              f" (L8-L12): depth changes {span(r['ip1'], '.3f')}, depth {span(r['ip0'])}")
    both = lambda key: [x for r in R.values() for x in r[key]]
    print(f"  both networks: rho {span(both('rho1'))}, Gamma {span(both('gA'))}, rings <= {max(both('gR')):.2f}"
          f" (L7-F1); depth probe rho {span(both('rho0'))} (L8-L12); tokens {span(both('tok'))} (L7-L12);"
          f" depth changes {span(both('ip1'), '.3f')}, depth {span(both('ip0'))} of the control's (L8-L12);"
          f" depth probes on the controls at L6-L7 at most {max(map(abs, both('ctl0'))):.3f} of exact (raw)")
    # Fig. 2c draws the depth-change probe from L6 and the depth probe from L8; before that, on the controls:
    pre = [(abs(Y[l][probe]["B"]["ratio"]), n, name, l, Y[l][probe]["B"].get("ratio_cal"))
           for n, Y in ((n, L["layers"]) for n, L in lay.items())
           for probe, name, last in (("oneform", "depth-change", "L5"), ("zeroform", "depth", "L7"))
           for l in between("L0", last)]
    top = max(pre)
    cal = [(abs(c), n, probe, l) for _, n, probe, l, c in pre if c is not None]
    print(f"  before the first layer Fig. 2c draws (depth-change probe L0-L5, depth probe L0-L7), every probe finds at"
          f" most {top[0]:.3f} of exact on the controls (raw; {top[1]} {top[2]} {top[3]})"
          + (f"; calibrated, where defined, at most {max(cal)[0]:.3f} ({' '.join(max(cal)[1:])})" if cal else ""))

    # ---------------------------------------------------------------- gluing
    G = json.load(open("results/summary_glue.json"))
    print("\ngluing, median r by junction weight (bar links weight 1): best weight and its margin over weight 1")
    gap = []
    for m, d in G.items():
        if "impossible" not in d:
            continue
        w = d["impossible"]["r_by_junction_weight"]
        best = max(w, key=w.get)
        gap.append(w[best] - w["1"])
        print(f"  {m:8s} best {best:>4s} (r {w[best]:.3f}), weight 1: r {w['1']:.3f}, margin {gap[-1]:.4f}")
    print(f"  {len(gap)} networks, weight 1 best or within {max(gap):.4f}")

    # ---------------------------------------------------------------- matched heads
    print("\nmatched heads on frozen DA-S (medians over images; Gamma and slopes as a ratio to exact):")
    for name, h in H.items():
        sA, sB = h['slope_ratio']['A'][0], h['slope_ratio']['B']
        print(f"  {name}: bar slope impossible {sA:.3f}, controls {sB:.3f}, impossible / controls {sA / sB:.2f} |"
              f" Gamma bars {h['gamma_bars']['A'][0]:.3f}, rings {h['gamma_bars']['rings']:+.3f} | one lap (path)"
              f" {h['gamma_path']['A']:.3f} | cut at k gives object k {h['cut_gives_object']:.2f} | loop closed on"
              f" {h['closed']['B']:.2f} of {h['n']['B']} controls | AUC |Gamma| vs controls {h['auc_vs_controls']:.2f}"
              + (f" | gap found {h['excess']['gap_found']:.2f}, largest e {h['excess']['max_e'][0]:+.2f}"
                 if "excess" in h else ""))
