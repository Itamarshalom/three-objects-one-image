"""Glue law: does least-squares gluing of a network's own readings of the opened loop predict its intact output?

The erasure probe (scripts/run_probe.py) gives, for every image, the intact output and three opened outputs (junction j
erased: cubes 5j-1, 5j, 5j+1 NaN). Each opened output reads the loop's edges i -> i+1 (i = 0..14, edge 14 closes the
loop) that it sees whole. Each output is put in its own units (divided by its mean |cube-to-cube change| on bar edges,
both cubes inside one bar: a "bar step"), and an edge's readings are averaged over the outputs that see it. Their sum
around the loop is the closure of the readings, in bar changes (5 bar steps): exact local geometry gives -3 on the
Penrose loop (the obstruction T) and 0 on a possible object.
Predictions of the intact loop profile (15 cubes):
  glue uniform / stiff : least-squares cube depths from the averaged readings, every edge weight 1 / the six junction
                         edges (touching a corner cube) weight STIFF; the closure residual is spread over every edge /
                         over the nine bar edges (exact readings: a flat loop / bars reversed at 0.66 of their slope).
                         On a loop this is the mean of the 15 objects cut at one edge each, edge i drawn with
                         probability ~ 1 / weight_i (Kirchhoff; checked below), while the mean of the objects cuts
                         only at the three gap edges
  object k             : opened output k completed into object k (gap at corner k): its three erased cubes integrated
                         from the visible ends with the averaged readings, its gap edge 5k-1 -> 5k left open
  mean of the objects  : the mean of the three completed objects (the Bayes mixture, in the network's own units)
  best single object   : the object closest to the intact output, chosen afterwards (an optimistic baseline)
  chosen single object : the object with the best median r for the same shading and chirality (perm, mirror) on the
                         other layout bins (a cross-validated single-object baseline)
Scores (impossible images):
  r                    : Pearson r with the intact profile over the 15 cubes (any scale and shift); median with a 95%
                         cluster-bootstrap interval over 15-degree layout bins; the share of images where gluing beats
                         the mean of the objects and the median paired difference; the median r over junction weights
  references           : floor = the network's typical intact profile for the same shading and chirality (median
                         z-scored intact profile, same perm and mirror, other layout bins); ceiling = the intact output
                         of the nearest other angle (same perm, mirror, texture and seed), a test-retest reliability
  tilt                 : mean change on bar edges / mean |change| over the loop (exact objects -0.54; > 0: the bars run
                         against the objects); medians, and the correlation across images of predicted and intact tilt,
                         also within layouts (both tilts minus their mean over the same perm, mirror and layout bin)
Samplers: each noise seed is one sample, its four conditions all from that seed; all seeds are pooled.
Sanity checks:
  exact geometry  : the exact objects (results/gt_profiles.csv), each under a random scale and shift, as the opened
                    outputs: the completed objects must be the objects, the closure -3, a possible object glued back;
                    and gluing random readings must equal the 1/weight-weighted mean of the 15 cut objects
  leave-one-out   : gluing a possible control's own opened outputs predicts its held-out intact output. Opening an
                    ordinary junction of a control leaves two pieces whose relative depth no cue fixes, so the reading
                    across the visible gap is unconstrained; also reported: integrating without that one reading, and
                    the gap in bar steps as the intact output makes it, as read, and as closing the chain needs
                    (exact 14)
usage: python scripts/analyze_glue.py [model ...]   (default: every results/probe_<model>.csv)
       -> results/summary_glue.json
"""
import os, sys, glob, json, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.analysis import load, boot_ci
from pdepth.probe import junction_cubes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_probe import ORDER, gt_profiles, corr

N, STIFF = 15, 100.0
P = [f"prof{i}" for i in range(N)]
JUNC = np.isin(np.arange(N), [(5 * j + d) % N for j in range(3) for d in (-1, 0)])  # edges i -> i+1 touching a corner
GAP = [(5 * k - 1) % N for k in range(3)]                    # object k's gap: edge 5k-1 -> 5k
DIFF = np.roll(np.eye(N), 1, 1) - np.eye(N)                  # row i: cube i+1 minus cube i
SWEEP = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]              # junction weights (bar edges 1)
PRED = ["glue_uniform", "glue_stiff", "mean_objects", "object_0", "object_1", "object_2", "best_object"]


def edges(x):
    return np.roll(x, -1, -1) - x


def readings(opened):
    """opened outputs [3, 15] (NaN on erased cubes) -> the outputs in their own units and the edge readings
    (bar steps) averaged over the outputs that see both cubes of the edge"""
    d = edges(opened)
    s = np.nanmean(np.abs(d[:, ~JUNC]), 1)
    return opened / s[:, None], np.nanmean(d / s[:, None], 0)


def glue(R, w_junc=1.0):
    """least-squares cube depths x with x[i+1] - x[i] ~ R[i] (junction edges weight w_junc, bar edges 1; edges without
    a reading dropped), mean 0; the closure residual is spread over the edges in proportion to 1 / weight"""
    ok = np.isfinite(R)
    sw = np.sqrt(np.where(JUNC, w_junc, 1.0)[ok])
    A = np.vstack([DIFF[ok] * sw[:, None], np.ones(N)])
    return np.linalg.lstsq(A, np.append(R[ok] * sw, 0.0), rcond=None)[0]


def complete(p, R, j):
    """opened output j (own units) completed into object j: erased cubes 5j-1, 5j, 5j+1 integrated from the visible
    ends with the readings R; the gap edge 5j-1 -> 5j stays open"""
    a, c, b = junction_cubes(j, N)
    o = p.copy()
    o[a] = o[a - 1] + R[a - 1]
    o[b] = o[(b + 1) % N] - R[b]
    o[c] = o[b] - R[c]
    return o


def tilt(x):
    d = edges(x)
    return d[~JUNC].mean() / np.abs(d).mean()


def opened_of(d):
    """the three opened outputs of one image [3, 15]; checks the erased cubes"""
    op = np.array([d[f"e{j}"] for j in range(3)], float)
    for j in range(3):
        assert set(np.where(~np.isfinite(op[j]))[0]) == set(junction_cubes(j, N)), f"e{j}: unexpected NaN pattern"
    return op


def score_impossible(d):
    intact = np.asarray(d["none"], float)
    p, R = readings(opened_of(d))
    objs = np.array([complete(p[j], R, j) for j in range(3)])
    pred = dict(glue_uniform=glue(R), glue_stiff=glue(R, STIFF), mean_objects=objs.mean(0),
                **{f"object_{k}": objs[k] for k in range(3)})
    o = {k: corr(v, intact) for k, v in pred.items()}
    o["best_object"] = max(o[f"object_{k}"] for k in range(3))
    o.update({f"glue_W{w:g}": corr(glue(R, w), intact) for w in SWEEP})
    o.update({f"tilt_{k}": tilt(pred[k]) for k in ("glue_uniform", "glue_stiff", "mean_objects")},
             tilt_intact=tilt(intact))
    o.update(closure=R.sum() / 5, read_bar=np.median(R[~JUNC]), read_junction=np.median(R[JUNC]),
             opened_vs_intact=np.mean([corr(p[j], intact) for j in range(3)]))
    return o


def score_control(d, k):
    intact = np.asarray(d["none"], float)
    p, R = readings(opened_of(d))
    R0 = R.copy(); R0[GAP[k]] = np.nan                       # without the reading across the visible gap
    objs = np.array([complete(p[j], R, j) for j in range(3)])
    gap = lambda x: edges(x)[GAP[k]] / np.abs(edges(x)[~JUNC]).mean()   # gap in the profile's own bar steps
    closing = R[GAP[k]] - R.sum()
    return dict(glue_uniform=corr(glue(R), intact), glue_stiff=corr(glue(R, STIFF), intact),
                integrate_without_gap=corr(glue(R0), intact), own_object=corr(objs[k], intact),
                mean_objects=corr(objs.mean(0), intact), closure=R.sum() / 5, gap_share=R[GAP[k]] / closing,
                gap_intact=gap(intact), gap_read=R[GAP[k]], gap_closing=closing, gap_glued=gap(glue(R)),
                opened_vs_intact_own=corr(p[k], intact),
                opened_vs_intact_other=np.mean([corr(p[j], intact) for j in range(3) if j != k]))


def references(A):
    """r of each impossible sample's intact output with the network's typical intact profile for its cube numbering
    (same perm and mirror, other layout bins) and with the intact output of the nearest other angle"""
    z = {i: (a["intact"] - a["intact"].mean()) / a["intact"].std() for i, a in enumerate(A)}
    same = lambda a: (a["perm"], a["mirror"], a["tex"], a["seed"], a["style"])
    tmpl, near = {}, defaultdict(list)
    for i, a in enumerate(A):
        near[same(a)].append(i)
    floor, ceil = [], []
    for i, a in enumerate(A):
        key = (a["perm"], a["mirror"], a["g"])
        if key not in tmpl:
            tmpl[key] = np.median([z[j] for j, b in enumerate(A)
                                   if (b["perm"], b["mirror"]) == key[:2] and b["g"] != key[2]], 0)
        floor.append(corr(tmpl[key], a["intact"]))
        others = [j for j in near[same(a)] if A[j]["angle"] != a["angle"]]
        ceil.append(corr(A[min(others, key=lambda j: abs(A[j]["angle"] - a["angle"]))]["intact"], a["intact"])
                    if others else np.nan)
    return np.array(floor), np.array(ceil)


def chosen_object(A):
    """r of the single object chosen for each sample's shading and chirality (perm, mirror) by its median r on the other
    layout bins"""
    lay = lambda a: (a["perm"], a["mirror"])
    pick = {}
    for a in A:
        key = lay(a) + (a["g"],)
        if key not in pick:
            tr = [[b[f"object_{k}"] for k in range(3)] for b in A if lay(b) == lay(a) and b["g"] != a["g"]]
            pick[key] = int(np.nanargmax(np.nanmedian(tr, 0))) if tr else None
    k = [pick[lay(a) + (a["g"],)] for a in A]
    return np.array([a[f"object_{j}"] if j is not None else np.nan for a, j in zip(A, k)]), k


def demean(x, keys):
    """x minus its mean over the samples with the same key"""
    x, keys = np.array(x, float), [str(k) for k in keys]
    for u in set(keys):
        s = np.array([k == u for k in keys]); x[s] -= np.nanmean(x[s])
    return x


def cut(R, i):
    """the object cut at edge i: cube depths integrated from the readings around the loop, starting at cube i+1"""
    return np.roll(np.r_[0.0, np.cumsum(np.roll(R, -(i + 1))[:-1])], i + 1)


def ci(x, groups, f=np.median):
    x, groups = np.asarray(x, float), np.asarray(groups)
    ok = np.isfinite(x)
    return [float(v) for v in boot_ci(x[ok], f, groups=groups[ok])]


def ci_corr(a, b, groups):
    """correlation across samples, cluster-bootstrap interval"""
    ok = np.isfinite(a) & np.isfinite(b)
    return [float(v) for v in boot_ci(np.c_[a, b][ok], lambda x: np.corrcoef(x.T)[0, 1], groups=np.asarray(groups)[ok])]


def summarise(rows):
    """rows of one probe table -> summary dict (None without complete impossible images)"""
    by = defaultdict(dict)
    for r in rows:
        if r["set"] in ("A", "B"):
            by[(r["stim"], r.get("seed", 0), r.get("style", ""), r.get("nuis", ""))][r["cond"]] = \
                (r, np.array([r[c] for c in P], float))
    A, B = [], []
    for (stim, seed, style, nuis), d in by.items():
        if not all(c in d for c in ("none", "e0", "e1", "e2")):
            continue
        r = d["none"][0]; prof = {c: v for c, (_, v) in d.items()}
        ang = int(r["angle"])
        rec = dict(stim=stim, seed=seed, style=(style, nuis), angle=ang, g=ang - ang % 15, perm=r.get("perm", ""),
                   mirror=r.get("mirror", ""), tex=r.get("tex", ""), intact=prof["none"])
        if r["set"] == "A":
            A.append(rec | score_impossible(prof))
        else:
            B.append(rec | score_control(prof, int(r["possible"])))
    if not A:
        return None
    col = lambda S, k: np.array([o[k] for o in S], float)
    gA = col(A, "g")
    imp = dict(n=len(A), n_images=len({a["stim"] for a in A}), n_seeds=len({a["seed"] for a in A}),
               r={k: ci(col(A, k), gA) for k in PRED})
    for name in ("uniform", "stiff"):
        x, m = col(A, f"glue_{name}"), col(A, "mean_objects")
        imp.setdefault("glue_beats_mean", {})[name] = ci((x > m).astype(float), gA, np.mean)
        imp.setdefault("glue_minus_mean", {})[name] = ci(x - m, gA)
        imp.setdefault("glue_beats_best_object", {})[name] = float(np.mean(x > col(A, "best_object")))
    rc, kc = chosen_object(A)
    imp["r"]["chosen_object"] = ci(rc, gA)
    imp["glue_beats_chosen_object"] = {name: ci((col(A, f"glue_{name}") > rc).astype(float), gA, np.mean)
                                       for name in ("uniform", "stiff")}
    imp["chosen_object_share"] = {str(j): float(np.mean([k == j for k in kc])) for j in range(3)}
    imp["r_by_junction_weight"] = {f"{w:g}": float(np.nanmedian(col(A, f"glue_W{w:g}"))) for w in SWEEP}
    floor, ceil = references(A)
    imp["references"] = dict(template=ci(floor, gA), nearest_angle=ci(ceil, gA),
                             glue_beats_template=ci((col(A, "glue_uniform") > floor).astype(float), gA, np.mean))
    ti = col(A, "tilt_intact")
    lays = [(a["perm"], a["mirror"], a["g"]) for a in A]
    imp["tilt"] = dict(median={k: float(np.nanmedian(col(A, f"tilt_{k}")))
                               for k in ("intact", "glue_uniform", "glue_stiff", "mean_objects")},
                       intact_reversed=float(np.mean(ti > 0)),
                       r_across_images={k: ci_corr(col(A, f"tilt_{k}"), ti, gA)
                                        for k in ("glue_uniform", "glue_stiff", "mean_objects")},
                       r_within_layouts={k: ci_corr(demean(col(A, f"tilt_{k}"), lays), demean(ti, lays), gA)
                                         for k in ("glue_uniform", "glue_stiff", "mean_objects")})
    imp["closure_bar_changes"] = ci(col(A, "closure"), gA)
    imp["reading_bar_steps"] = dict(bar=float(np.median(col(A, "read_bar"))),
                                    junction=float(np.median(col(A, "read_junction"))))
    imp["opened_vs_intact"] = float(np.nanmedian(col(A, "opened_vs_intact")))
    out = {"impossible": imp}
    if B:
        gB = col(B, "g")
        ks = ["glue_uniform", "glue_stiff", "integrate_without_gap", "own_object", "mean_objects"]
        out["control_loo"] = dict(n=len(B), r={k: ci(col(B, k), gB) for k in ks},
                                  closure_bar_changes=ci(col(B, "closure"), gB), gap_share=ci(col(B, "gap_share"), gB),
                                  gap_bar_steps={k: float(np.median(col(B, f"gap_{k}")))
                                                 for k in ("intact", "closing", "read", "glued")},
                                  opened_vs_intact=dict(own=float(np.nanmedian(col(B, "opened_vs_intact_own"))),
                                                        other=float(np.nanmedian(col(B, "opened_vs_intact_other")))))
    return out


def exact_check(stims, G, seed=0):
    """exact objects as the outputs (every opened output under its own random scale and shift)"""
    rng = np.random.default_rng(seed)
    aff = lambda x: x * rng.uniform(0.3, 3.0) + rng.uniform(-10, 10)
    hole = lambda j: np.isin(np.arange(N), junction_cubes(j, N))
    obj_err, mean_r, clo_A, slope, glue_B, clo_B = [], [], [], defaultdict(list), [], []
    for s in sorted(stims):
        g = G.get(s)
        if g is None or not np.isfinite(g).all():
            continue
        if s[0] == "A":
            p, R = readings(np.array([aff(np.where(hole(j), np.nan, g[j])) for j in range(3)]))
            objs = np.array([complete(p[j], R, j) for j in range(3)])
            obj_err.append(max(1 - corr(objs[j], g[j]) for j in range(3)))
            mean_r.append(corr(objs.mean(0), g.mean(0)))
            clo_A.append(R.sum() / 5)
            for name, w in (("uniform", 1.0), ("stiff", STIFF)):          # glued bar change / read bar change
                slope[name].append(edges(glue(R, w))[~JUNC].mean() / R[~JUNC].mean())
        elif s[0] == "B":
            k = int(s.split("_k")[1][0])
            p, R = readings(np.array([aff(np.where(hole(j), np.nan, g[k])) for j in range(3)]))
            glue_B.append(min(corr(glue(R, w), g[k]) for w in (1.0, STIFF)))
            clo_B.append(R.sum() / 5)
    R, kir = rng.normal(-1, 0.5, N), []                     # Kirchhoff: glue = mean of the cut objects, cut ~ 1/weight
    for w in SWEEP:
        q = 1 / np.where(JUNC, w, 1.0); q /= q.sum()
        m = sum(q[i] * cut(R, i) for i in range(N))
        kir.append(np.abs(m - m.mean() - glue(R, w)).max())
    return dict(n_impossible=len(obj_err), n_control=len(glue_B), glue_vs_mean_of_cuts_max_abs=float(np.max(kir)),
                completed_objects_max_1_minus_r=float(np.max(obj_err)), mean_objects_min_r=float(np.min(mean_r)),
                closure_impossible=[float(np.min(clo_A)), float(np.max(clo_A))],
                glued_bar_slope_of_object_slope={k: float(np.median(v)) for k, v in slope.items()},
                glued_bar_slope_theory={"uniform": 0.0, "stiff": float(1 - 15 / (9 + 6 / STIFF))},
                control_glue_min_r=float(np.min(glue_B)), closure_control_max_abs=float(np.max(np.abs(clo_B))))


def fmt(v, d=2):
    return f"{v[0]:+.{d}f} [{v[1]:+.{d}f},{v[2]:+.{d}f}]"


if __name__ == "__main__":
    files = {os.path.basename(f)[6:-4]: f for f in glob.glob("results/probe_*.csv")
             if not any(t in os.path.basename(f) for t in ("loops", "attn"))}
    models = sys.argv[1:] or [m for m in ORDER if m in files] + sorted(set(files) - set(ORDER))
    S, stims = {}, set()
    print(f"{'network':8s} {'n':>5s}  {'glue uniform':20s} {'glue stiff':20s} {'mean of objects':20s} {'objects 0/1/2':16s} "
          f"{'best obj':8s} | {'glue > mean (uniform, stiff)':36s}")
    for m in models:
        rows = [r for r in load(files[m]) if r.get("seed", 0) != -1]       # -1: seed-averaged rows, if any
        stims |= {r["stim"] for r in rows if r["set"] in ("A", "B")}
        res = summarise(rows)
        if res is None:
            continue
        S[m] = res
        a = res["impossible"]; r = a["r"]
        print(f"{m:8s} {a['n']:5d}  {fmt(r['glue_uniform']):20s} {fmt(r['glue_stiff']):20s} {fmt(r['mean_objects']):20s} "
              f"{r['object_0'][0]:+.2f}/{r['object_1'][0]:+.2f}/{r['object_2'][0]:+.2f}  {r['best_object'][0]:+.2f}    | "
              f"{fmt(a['glue_beats_mean']['uniform'])}, {fmt(a['glue_beats_mean']['stiff'])}")
        rf, t = a["references"], a["tilt"]
        print(f"{'':16s}paired r gain over the mean of objects: uniform {fmt(a['glue_minus_mean']['uniform'])}, stiff "
              f"{fmt(a['glue_minus_mean']['stiff'])}; glue beats the best object (post hoc) "
              f"{a['glue_beats_best_object']['uniform']:.2f} / {a['glue_beats_best_object']['stiff']:.2f}")
        cs = a["chosen_object_share"]
        print(f"{'':16s}single object chosen per shading and chirality on the other bins (0/1/2 picked {cs['0']:.2f}/"
              f"{cs['1']:.2f}/{cs['2']:.2f}): r {fmt(r['chosen_object'])}; glue beats it: uniform "
              f"{fmt(a['glue_beats_chosen_object']['uniform'])}, stiff {fmt(a['glue_beats_chosen_object']['stiff'])}")
        print(f"{'':16s}references: typical intact profile {fmt(rf['template'])} (glue beats it "
              f"{rf['glue_beats_template'][0]:.2f}), intact at the nearest other angle {fmt(rf['nearest_angle'])}")
        print(f"{'':16s}r by junction weight: " + ", ".join(f"{w}: {v:.2f}" for w, v in a["r_by_junction_weight"].items()))
        print(f"{'':16s}tilt: intact {t['median']['intact']:+.2f} (reversed {t['intact_reversed']:.2f}), glue uniform "
              f"{t['median']['glue_uniform']:+.2f}, stiff {t['median']['glue_stiff']:+.2f}, mean of objects "
              f"{t['median']['mean_objects']:+.2f}; across images r with intact: uniform "
              f"{fmt(t['r_across_images']['glue_uniform'])}, stiff {fmt(t['r_across_images']['glue_stiff'])}, mean of "
              f"objects {fmt(t['r_across_images']['mean_objects'])}")
        print(f"{'':16s}tilt r within layouts (same perm, mirror and bin): uniform {fmt(t['r_within_layouts']['glue_uniform'])}"
              f", stiff {fmt(t['r_within_layouts']['glue_stiff'])}, mean of objects {fmt(t['r_within_layouts']['mean_objects'])}")
        print(f"{'':16s}closure of the readings {fmt(a['closure_bar_changes'])} bar changes (exact -3); readings "
              f"bar {a['reading_bar_steps']['bar']:+.2f}, junction {a['reading_bar_steps']['junction']:+.2f} bar steps "
              f"(exact -1, -1); opened vs intact r {a['opened_vs_intact']:.2f}")
        if "control_loo" in res:
            c = res["control_loo"]; r = c["r"]; gp = c["gap_bar_steps"]
            print(f"{'':16s}controls, intact held out (n={c['n']}): glue uniform {fmt(r['glue_uniform'])}, stiff "
                  f"{fmt(r['glue_stiff'])}, without the gap reading {fmt(r['integrate_without_gap'])}, own object "
                  f"{fmt(r['own_object'])}, mean of objects {fmt(r['mean_objects'])}")
            print(f"{'':16s}  closure {fmt(c['closure_bar_changes'])} bar changes (exact 0); gap in bar steps: intact "
                  f"{gp['intact']:+.1f}, closing needs {gp['closing']:+.1f}, read {gp['read']:+.1f} "
                  f"(share {fmt(c['gap_share'])}), glued {gp['glued']:+.1f} (exact +14); opened vs intact r: gap "
                  f"erased {c['opened_vs_intact']['own']:.2f}, another junction erased {c['opened_vs_intact']['other']:.2f}")
    with np.errstate(invalid="ignore", divide="ignore"):
        S["exact_geometry"] = ex = exact_check(stims, gt_profiles())
    print(f"exact geometry ({ex['n_impossible']} impossible, {ex['n_control']} controls): completed objects 1 - r <= "
          f"{ex['completed_objects_max_1_minus_r']:.1e}, mean of objects r >= {ex['mean_objects_min_r']:.6f}, closure "
          f"{ex['closure_impossible'][0]:.3f}..{ex['closure_impossible'][1]:.3f} bar changes; glued bar change / read bar "
          f"change: uniform {ex['glued_bar_slope_of_object_slope']['uniform']:+.3f}, stiff "
          f"{ex['glued_bar_slope_of_object_slope']['stiff']:+.3f} (theory 0, {ex['glued_bar_slope_theory']['stiff']:+.3f})"
          f" | controls glued back r >= {ex['control_glue_min_r']:.6f}, |closure| <= {ex['closure_control_max_abs']:.3f}"
          f" | glue = weighted mean of the 15 cuts, max |diff| {ex['glue_vs_mean_of_cuts_max_abs']:.1e}")
    json.dump(S, open("results/summary_glue.json", "w"), indent=1, default=float)
