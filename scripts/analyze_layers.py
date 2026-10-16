"""Layer-wise linear read-outs of a Depth Anything V2 network's frozen features (scripts/run_features.py): along the
bars of the impossible figure, does a 1-form read-out (local depth change) run with the objects where a 0-form
read-out (depth itself) bends?

Layers: patch tokens after the embeddings and after each ViT block (L0..L12 for DA-S/B) and the four DPT fusion maps
(F1 fuses the deepest tap, F4 feeds the head), plus the network's own output (OUT, dep_grid, read as a 0-form).
Two ridge probes per layer, fit ONLY on the training scenes (possible cube chains, data/scenes; outputs scene_*.npz):
  1-form: features -> exact per-patch depth gradient g (dimensionless), patches with face-interior coverage >= 0.5
  0-form: features -> exact per-patch depth (pixels), patches >= 75% figure; features and depth centred per image
          (OUT: one global scale on the centred output)
Ridge in closed form from per-scene sufficient statistics (standardised features): alpha by 5-fold CV over 80% of the
scenes (folds = whole scenes); on the other 20% (held out): R2, the shrinkage s = slope of prediction on target, and
the slope gain k = the same along-run slope read-out as below on the scenes' straight runs (chains regenerated from
the scene seeds), pooled regression of predicted on exact run slope; calibrated values are raw / k.
On the impossible images (set A), their possible controls (set B) and the closed possible rings (data/loops):
  along-bar slope of bar j (travel corner j -> j+1, interior cubes 5j+1..5j+4), exact -1/sqrt(2) (rings: +-):
    1-form: mean of predicted g . t over the bar's side-face patches
    0-form: the paper's loop read-out (pdepth/loop.py) on the predicted depth map: upsampled bilinearly from the patch
            centres to the stimulus pixels, median per interior cube (eroded masks as in pdepth/readout.py), line
            through the four cubes, per cube step -> per pixel
  reported as ratio to exact (raw and calibrated), share of bars running with the objects, and rho = impossible /
  the controls of the same 15-degree layout bin and chirality (median, cluster-bootstrap CI over layout bins)
  loop integral (bars only, corner to corner): Gamma = sum_j slope_j x bar length = sqrt(2)/3 sum_j slope_j (units of
    the obstruction T), calibrated (exact -1 on A and on B's bars, 0 on the rings); AUC A vs rings of |Gamma| (any
    non-closure) and of -Gamma (the obstruction with its sign)
  identical pixels: A_a*_p0_t0 and its control B_k differ only near corner k; on the bars away from it (side-face
    patches, and whole cubes whose read-out uses no patch within 2 patches of a differing pixel): both probes' slopes
    on A vs B_k, and the relative change of the features there
Layers whose slope gain k is below 0.1 (no slope decodable) get no calibrated values and no rho in the table.
Also saved: the probes and the per-image slopes (outputs/layers_<model>.npz).
usage: python scripts/analyze_layers.py [dav2-s|dav2-b] [--scenes N] [--patches K]   (first N training scenes, at most
       K random patches per scene and probe; default all)   -> results/summary_layers_<model>.json
"""
import os, sys, re, glob, json, time, warnings, numpy as np
from collections import defaultdict
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pdepth.analysis import boot_ci
from pdepth.loop import bars_and_joints
from pdepth.normals import gt_normals, bar_directions, PX
from pdepth.readout import ERODE
from pdepth.scenes import random_scene

G = 37
IDX = np.floor((np.arange(518) + 0.5) * 512 / 518).astype(int)          # 518-px input pixel -> 512-px stimulus pixel
GC = (np.arange(512) + 0.5) * 518 / 512 / 14 - 0.5                     # stimulus pixel centre -> patch-grid coordinate
S2 = 1 / np.sqrt(2)                                                      # |along-bar slope| of every bar
LT = np.sqrt(2) / 3                                                      # bar length (5 cube steps) / T
ALPHAS = 10.0 ** np.arange(-6, 1.01, 0.5)
GCOV, FCOV, FBAR, DIL = 0.5, 0.75, 0.9, 2
NFOLD, CHUNK, KMIN = 5, 16, 0.1                                         # no calibration below a slope gain of KMIN


# ---------------------------------------------------------------- features and ridge from sufficient statistics
def layer_feats(d, nl, nf):
    """[n_keep, D] float32 per read-out layer: hidden states, fusion maps, the network output (last)"""
    hs, fus = d["hs"].astype(np.float32), d["fus"].astype(np.float32)
    return [hs[i] for i in range(nl)] + [fus[i] for i in range(nf)] + [d["dep_grid"].reshape(-1)[d["keep"]][:, None]]


def suff(X, Y):
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    return [len(X), X.sum(0), X.T @ X, X.T @ Y, Y.sum(0), Y.T @ Y]


def add(a, b, sign=1):
    return b if a is None else [x + sign * y for x, y in zip(a, b)]


def solve(st, alphas):
    """ridge on standardised features (penalty alpha * n); returns the affine map (A [D, q], c [q]) per alpha"""
    n, sx, sxx, sxy, sy, _ = st
    mu, my = sx / n, sy / n
    C = sxx / n - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(C), 0)) + 1e-6
    lam, V = np.linalg.eigh(C / np.outer(sd, sd))
    VC = V.T @ ((sxy / n - np.outer(mu, my)) / sd[:, None])
    out = []
    for a in alphas:
        A = (V @ (VC / (np.maximum(lam, 0) + a)[:, None])) / sd[:, None]
        out.append((A, my - mu @ A))
    return out


def score(st, A, c):
    """SSE, SST (both pooled over outputs) and the slope of prediction on target, from sufficient statistics"""
    n, sx, sxx, sxy, sy, syy = st
    xa = sx @ A
    spp = A.T @ sxx @ A + np.outer(c, xa) + np.outer(xa, c) + n * np.outer(c, c)
    spy = A.T @ sxy + np.outer(c, sy)
    my = sy / n
    sst = np.trace(syy) - n * my @ my
    return np.trace(syy) - 2 * np.trace(spy) + np.trace(spp), sst, (np.trace(spy) - (xa + n * c) @ my) / sst


def scene_rows(d, nl, nf, cap=None, seed=0):
    """training rows of one scene, per layer: 1-form (X, g) and 0-form (X, depth) centred per image
    (cap: at most that many random patches per probe)"""
    keep = d["keep"]
    i1 = np.where(d["gcov"].reshape(-1)[keep] >= GCOV)[0]
    i0 = np.where(d["fcov"].reshape(-1)[keep] >= FCOV)[0]
    if cap:
        rng = np.random.default_rng(seed)
        i1, i0 = (np.sort(rng.choice(i, min(cap, len(i)), replace=False)) for i in (i1, i0))
    g = d["grad"].reshape(-1, 2)[keep][i1]
    y = d["dgrid"][0].reshape(-1)[keep][i0][:, None].astype(np.float64)
    y = y - y.mean()
    return [(X[i1], g, X[i0] - X[i0].mean(0), y) for X in layer_feats(d, nl, nf)]


def seed_of(fn):
    return int(re.findall(r"\d+", os.path.basename(fn))[0])


def fit_probes(scenes, nl, nf, cap=None):
    """CV over alphas, refit on all CV folds, held-out R2 and shrinkage; probe '1' for layers < nl+nf, '0' for all"""
    nL = nl + nf + 1
    ST = defaultdict(lambda: None)                                      # (probe, layer, group) -> statistics
    own = np.zeros(2)                                                   # OUT with a per-image scale (held-out scenes)
    by = defaultdict(list)                                              # held-out: index % 5 == 0; CV folds: the rest
    for fn in scenes:
        i = seed_of(fn)
        by[NFOLD if i % 5 == 0 else (i // 5) % NFOLD].append(fn)
    t0 = time.time()
    for gi, fs in sorted(by.items()):
        for ch in range(0, len(fs), CHUNK):
            rows = [scene_rows(np.load(fn), nl, nf, cap, seed_of(fn)) for fn in fs[ch:ch + CHUNK]]
            for li in range(nL):
                if li < nL - 1:
                    ST["1", li, gi] = add(ST["1", li, gi], suff(np.concatenate([r[li][0] for r in rows]),
                                                                np.concatenate([r[li][1] for r in rows])))
                ST["0", li, gi] = add(ST["0", li, gi], suff(np.concatenate([r[li][2] for r in rows]),
                                                            np.concatenate([r[li][3] for r in rows])))
            if gi == NFOLD:
                for r in rows:
                    x, y = r[-1][2][:, 0].astype(np.float64), r[-1][3][:, 0]
                    own += [(y * y).sum() - (x @ y) ** 2 / max(x @ x, 1e-12), (y * y).sum()]
        print(f"  scene group {gi}: {len(fs)} scenes ({time.time() - t0:.0f}s)", flush=True)
    fits = {}
    for p in ("1", "0"):
        for li in range(nL - (p == "1")):
            cv = [ST[p, li, g] for g in range(NFOLD)]
            tot = None
            for s in cv:
                tot = add(tot, s)
            sse, sst = np.zeros(len(ALPHAS)), np.zeros(len(ALPHAS))
            for s in cv:
                for ai, (A, c) in enumerate(solve(add(tot, s, -1), ALPHAS)):
                    e, t, _ = score(s, A, c)
                    sse[ai] += e; sst[ai] += t
            r2cv = 1 - sse / sst
            ai = int(np.argmax(r2cv))
            A, c = solve(tot, [ALPHAS[ai]])[0]
            e, t, sh = score(ST[p, li, NFOLD], A, c)
            fits[p, li] = dict(A=A, c=c, alpha=float(ALPHAS[ai]), r2_cv=float(r2cv[ai]), r2=float(1 - e / t),
                               shrink=float(sh), n_fit=int(tot[0]), n_test=int(ST[p, li, NFOLD][0]))
    fits["0", nL - 1]["r2_scale_per_image"] = float(1 - own[0] / own[1])
    return fits, by


# ---------------------------------------------------------------- slopes along bars and runs
def slope1(P, i, t):
    """mean of g . t over patches i, for every layer of P [L, G*G, 2]"""
    return (P[:, i] @ t).mean(1) if len(i) >= 3 else np.full(len(P), np.nan)


def side_patches(d, cubes, t, sgn):
    """patches of the given cubes on side faces: face interiors where the exact g . t = sgn/sqrt2 (sgn 0: either)"""
    on = d["keep"] & np.isin(d["maj"].reshape(-1), cubes) & (d["gcov"].reshape(-1) >= GCOV)
    gt = d["grad"].reshape(-1, 2) @ t
    return np.where(on & (np.abs(gt - sgn * S2) < 0.05 if sgn else np.abs(np.abs(gt) - S2) < 0.05))[0]


def cube_pixels(cube, cubes):
    """pixels of the given cubes (eroded as in pdepth/readout.py), each cube's rows, and the bilinear weights
    [n_pix, G*G] from the patch centres to them"""
    rows, cols, groups, n = [], [], [], 0
    for c in cubes:
        rr, cc = np.nonzero(cube == c)
        if len(rr):
            r0, c0 = max(rr.min() - 4, 0), max(cc.min() - 4, 0)
            m = cube[r0:rr.max() + 5, c0:cc.max() + 5] == c
            for e in (ERODE, 1, 0):
                me = binary_erosion(m, iterations=e) if e else m
                if me.sum() >= 20 or e == 0:
                    break
            rr, cc = np.nonzero(me)
            rr, cc = rr + r0, cc + c0
        rows.append(rr); cols.append(cc); groups.append(np.arange(n, n + len(rr))); n += len(rr)
    rr, cc = np.concatenate(rows), np.concatenate(cols)
    r, c = np.clip(GC[rr], 0, G - 1), np.clip(GC[cc], 0, G - 1)
    r0, c0 = np.minimum(r.astype(int), G - 2), np.minimum(c.astype(int), G - 2)
    fr, fc = r - r0, c - c0
    idx = np.stack([r0 * G + c0, r0 * G + c0 + 1, (r0 + 1) * G + c0, (r0 + 1) * G + c0 + 1], 1)
    w = np.stack([(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc], 1)
    return dict(rr=rr, cc=cc, groups=groups,
                W=csr_matrix((w.ravel(), (np.repeat(np.arange(n), 4), idx.ravel())), shape=(n, G * G)))


def cube_values(P, cp):
    """per-cube median of the grid maps P [L, G*G], upsampled bilinearly (NaN-aware) to the cubes' pixels -> [L, n]"""
    ok = np.isfinite(P)
    with np.errstate(invalid="ignore", divide="ignore"):
        V = (cp["W"] @ np.where(ok, P, 0).T) / (cp["W"] @ ok.T.astype(float))
    return np.stack([np.nanmedian(V[g], 0) if len(g) else np.full(len(P), np.nan) for g in cp["groups"]], 1)


def slope0(V, step_px):
    """slope per pixel along the direction of travel of per-cube values V [L, n] (consecutive cubes, NaN = unused)"""
    ok = np.isfinite(V).all(0)
    if ok.sum() < 2:
        return np.full(len(V), np.nan)
    return np.polyfit(np.arange(V.shape[1])[ok], V[:, ok].T, 1)[0] / step_px


def geometry(d, st):
    """bar directions t (x right, y up), exact slope signs, per bar the 1-form patches, the interior cubes' pixels
    (bar j: groups 4j..4j+3) and the length of one cube step in pixels"""
    nb = int(d["n_bars"]) if "n_bars" in d else 3
    cube = st["cube"]
    if "corner_xy" in st:                                                # Penrose stimuli: exact corners
        t, sg = bar_directions(st), -np.ones(3)
    else:                                                                # rings: centroid direction snapped to an axis
        n, ok = gt_normals(st, float(st["px"]))
        u, cnt = np.unique(np.round(n[ok][:, :2] / n[ok][:, 2:], 2), axis=0, return_counts=True)
        u = u[cnt > 0.01 * cnt.sum()]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        t = []
        for j in range(nb):
            a, b = (np.argwhere(cube == c).mean(0) for c in (5 * j + 1, 5 * j + 4))
            dd = np.array([b[1] - a[1], a[0] - b[0]])
            m = np.argmax(np.abs(u @ dd))
            t.append(u[m] * np.sign(u[m] @ dd))
        t = np.array(t)
        prof = np.array([np.nanmedian(st["depth"][0][cube == c]) for c in range(int(st["n_cubes"]))])
        sg = np.sign(bars_and_joints(prof, nb)[0])
    side = [side_patches(d, np.arange(5 * j + 1, 5 * j + 5), t[j], sg[j]) for j in range(nb)]
    cp = cube_pixels(cube, [5 * j + q for j in range(nb) for q in range(1, 5)])
    return t, sg, side, cp, np.sqrt(2 / 3) / float(st.get("px", PX))


def scene_runs(d, sc, seed):
    """straight runs (>= 2 interior cubes) of a training scene, its cube chains regenerated from the seed: image
    direction of travel, the interior cubes' side-face patches and pixels"""
    pos, _ = random_scene(np.random.default_rng(seed))
    if len(pos) != int(sc["n_cubes"]):
        return []
    face, grad, cube = sc["face"], sc["grad"], sc["cube"]
    gface = [grad[:, face == a].mean(1) if (face == a).sum() > 50 else None for a in range(3)]   # || image of +e_a
    step = np.diff(pos, axis=0)
    out, i = [], 0
    while i < len(step):
        j = i
        while j + 1 < len(step) and (step[j + 1] == step[i]).all():
            j += 1
        a = int(np.argmax(np.abs(step[i])))
        if np.abs(step[i]).sum() == 1 and j - i >= 2 and gface[a] is not None:
            t = np.sign(step[i][a]) * gface[a] / np.linalg.norm(gface[a])
            cubes = np.arange(i + 1, j + 1)
            out.append((t, side_patches(d, cubes, t, 0), cube_pixels(cube, cubes)))
        i = j + 1
    return out


def predict(d, fits, nl, nf):
    """1-form [nl+nf, G*G, 2] and 0-form [nl+nf+1, G*G] read-outs on the full grid (NaN off the kept patches)"""
    keep = d["keep"]
    i0 = d["fcov"].reshape(-1)[keep] >= FCOV
    F = layer_feats(d, nl, nf)
    P1 = np.full((len(F) - 1, G * G, 2), np.nan)
    P0 = np.full((len(F), G * G), np.nan)
    for li, X in enumerate(F):
        if li < len(F) - 1:
            P1[li, keep] = X @ fits["1", li]["A"] + fits["1", li]["c"]
        P0[li, keep] = ((X - X[i0].mean(0)) @ fits["0", li]["A"] + fits["0", li]["c"])[:, 0]
    return P1, P0


def exact_map(d, k):
    """exact per-patch depth of object k [1, G*G], masked like the probes' maps (dgrid is 0, not NaN, off the figure)"""
    return np.where(d["keep"], d["dgrid"][k].reshape(-1), np.nan)[None]


def run_gain(test, fits, nl, nf):
    """slope gain k of each probe on the held-out scenes' straight runs: sum pred * exact / sum exact^2 over the runs
    whose exact read-out is within 20% of 1/sqrt2 (drops runs cut by occlusion)"""
    acc = defaultdict(list)
    for fn in test:
        d = dict(np.load(fn))
        sc = np.load(os.path.join(ROOT, "data/scenes", os.path.basename(fn)))
        runs = scene_runs(d, sc, seed_of(fn))
        if not runs:
            continue
        P1, P0 = predict(d, fits, nl, nf)
        g, y, step = d["grad"].reshape(1, -1, 2), exact_map(d, 0), np.sqrt(2 / 3) / float(sc["px"])
        for t, side, cp in runs:
            acc["1"].append(np.r_[slope1(P1, side, t), slope1(g, side, t)])
            acc["0"].append(slope0(cube_values(np.vstack([P0, y]), cp), step))
    ex = {}
    for p, v in acc.items():
        v = np.array(v)
        n_all = len(v)
        v = v[np.isfinite(v).all(1) & (np.abs(np.abs(v[:, -1]) / S2 - 1) < 0.2)]
        P, E = v[:, :-1], v[:, -1]
        for li in range(P.shape[1]):
            fits[p, li].update(gain=float((P[:, li] * E).sum() / (E ** 2).sum()), n_runs=len(v),
                               runs_right=float(np.mean(np.sign(P[:, li]) == np.sign(E))))
        ex[p] = dict(n_runs=len(v), n_runs_all=n_all, exact_abs_ratio=float(np.mean(np.abs(E)) / S2),
                     exact_abs_sd=float(np.std(np.abs(E)) / S2))
    return ex


def evaluate(d, st, fits, nl, nf):
    t, sg, side, cp, step = geometry(d, st)
    nb = len(t)
    P1, P0 = predict(d, fits, nl, nf)
    k = int(d["possible"]) if "possible" in d and int(d["possible"]) >= 0 else 0
    V = cube_values(np.vstack([P0, exact_map(d, k)]), cp)                               # [L + 1, 4 nb]
    full = np.array([np.median(d["dep"][cp["rr"][g], cp["cc"][g]]) if len(g) else np.nan for g in cp["groups"]])
    s0 = np.stack([slope0(V[:, 4 * j:4 * j + 4], step) for j in range(nb)], 1)        # [L + 1, nb]
    return dict(t=t, sg=sg, side=side, cp=cp, step=step, P1=P1, P0=P0, s0=s0[:-1], ex0=s0[-1],
                s1=np.stack([slope1(P1, side[j], t[j]) for j in range(nb)], 1),
                ex1=np.array([slope1(d["grad"].reshape(1, -1, 2), side[j], t[j])[0] for j in range(nb)]),
                out_full=np.array([slope0(full[None, 4 * j:4 * j + 4], step)[0] for j in range(nb)]),
                nside=[len(i) for i in side])


# ---------------------------------------------------------------- summaries
def lay(m):
    return (m["angle"] - m["angle"] % 15, m["mirror"])


def auc(pos, neg):
    pos, neg = np.asarray(pos)[np.isfinite(pos)], np.asarray(neg)[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def summarise(R, meta, key, li, k):
    """one probe at one layer: ratios, shares, rho, loop integral over sets A, B and the rings (k = slope gain)"""
    out = {}
    ratio = {n: np.nanmean(R[n][key][li] / (R[n]["sg"] * S2)) for n in R}           # per image, raw
    graw = {n: LT * np.sum(R[n][key][li]) for n in R}                                 # bars only, / T
    k = k if k >= KMIN else np.nan
    fam = {"A": [n for n in R if meta[n]["set"] == "A"], "B": [n for n in R if meta[n]["set"] == "B"],
           "rings": [n for n in R if meta[n]["set"] in ("H", "R")]}
    for S, ns in fam.items():
        if not ns:
            continue
        r = np.array([ratio[n] for n in ns])
        bars = np.concatenate([R[n][key][li] * R[n]["sg"] for n in ns])
        gg = np.array([graw[n] / k for n in ns])
        grp = np.array([lay(meta[n])[0] if S != "rings" else meta[n]["angle"] for n in ns])
        ci = boot_ci(r[np.isfinite(r)], np.median, n=1000, groups=grp[np.isfinite(r)])
        out[S] = dict(n=len(ns), ratio=f(ci[0]), ratio_ci=[f(ci[1]), f(ci[2])], ratio_cal=f(ci[0] / k),
                      with_objects=f(np.mean(bars[np.isfinite(bars)] > 0)),
                      all_bars=f(np.mean([np.all(R[n][key][li] * R[n]["sg"] > 0) for n in ns])),
                      gamma=f(np.nanmedian(gg)), gamma_iqr=[f(np.nanpercentile(gg, 25)), f(np.nanpercentile(gg, 75))],
                      gamma_raw=f(np.nanmedian([graw[n] for n in ns])),
                      n_nan_bars=int(np.sum(~np.isfinite(bars))))
    if "A" in out and "B" in out:
        ctl = defaultdict(list)
        for n in fam["B"]:
            ctl[lay(meta[n])].append(ratio[n])
        rho = [(ratio[n] / np.nanmean(ctl[lay(meta[n])]), lay(meta[n])[0]) for n in fam["A"] if lay(meta[n]) in ctl]
        rho = [(x, g) for x, g in rho if np.isfinite(x)]
        ci = boot_ci([x for x, _ in rho], np.median, n=1000, groups=[g for _, g in rho])
        mB = np.nanmean([ratio[n] for n in fam["B"]])
        out["A"].update(rho=f(ci[0]), rho_ci=[f(ci[1]), f(ci[2])], n_rho=len(rho),
                        rho_pooled=f(np.nanmean([ratio[n] for n in fam["A"]]) / mB))
    if "A" in out and "rings" in out:
        out["auc_gamma_A_vs_rings"] = f(auc([abs(graw[n]) for n in fam["A"]], [abs(graw[n]) for n in fam["rings"]]))
        out["auc_signed_A_vs_rings"] = f(auc([-graw[n] for n in fam["A"]], [-graw[n] for n in fam["rings"]]))
    return out


def identical_pixels(R, feats_dir, nl, nf, nL):
    """A_a*_p0_t0 vs B_a*_k (same layout and chirality): read-outs on the bars away from every differing pixel"""
    pairs = []
    for a in R:
        m = re.fullmatch(r"A_a(\d+)_p0_t0(_m)?", a)
        for k in range(3):
            b = f"B_a{m.group(1)}_k{k}{m.group(2) or ''}" if m else None
            if b in R:
                pairs.append((a, b, k))
    if not pairs:
        return None
    acc = defaultdict(list)
    interior = [5 * j + q for j in range(3) for q in range(1, 5)]
    for a, b, k in pairs:
        ia, ib = (np.load(os.path.join(ROOT, "data/stim2", n + ".npz"))["img"] for n in (a, b))
        diff = (ia != ib).any(-1)[IDX][:, IDX].reshape(G, 14, G, 14).any((1, 3))
        near = binary_dilation(diff, iterations=DIL).reshape(-1)
        Ra, Rb, cp = R[a], R[b], R[a]["cp"]                     # cubes away from the difference: same pixels in A, B_k
        use = np.array([len(g) > 0 and (cp["W"][g] @ near.astype(float)).max() == 0 for g in cp["groups"]])
        Va, Vb = cube_values(Ra["P0"], cp), cube_values(Rb["P0"], cp)
        Va[:, ~use], Vb[:, ~use] = np.nan, np.nan
        for j in range(3):
            t, q = Ra["t"][j], slice(4 * j, 4 * j + 4)
            i1 = np.intersect1d(Ra["side"][j], Rb["side"][j])
            i1 = i1[~near[i1]]
            acc["s1"].append(np.stack([slope1(Ra["P1"], i1, t), slope1(Rb["P1"], i1, t)], 1))     # [L, 2]
            acc["s0"].append(np.stack([slope0(Va[:, q], Ra["step"]), slope0(Vb[:, q], Ra["step"])], 1))
            acc["n"].append((len(i1), use[q].sum()))
        da, db = np.load(os.path.join(feats_dir, a + ".npz")), np.load(os.path.join(feats_dir, b + ".npz"))
        i = np.where(da["keep"] & db["keep"] & ~near & np.isin(da["maj"].reshape(-1), interior) &
                     (da["fcov"].reshape(-1) >= FBAR))[0]
        pa, pb = np.cumsum(da["keep"])[i] - 1, np.cumsum(db["keep"])[i] - 1
        acc["tok"].append([np.linalg.norm(Xa[pa] - Xb[pb], axis=1).mean() /
                           (np.linalg.norm(Xb[pb] - Xb[pb].mean(0), axis=1).mean() + 1e-9)
                           for Xa, Xb in zip(layer_feats(da, nl, nf), layer_feats(db, nl, nf))])
        acc["ndiff"].append(int(diff.sum()))
    out = dict(n_pairs=len(pairs), n_bars=3 * len(pairs), patches_differing=f(np.mean(acc["ndiff"])),
               side_patches_per_bar=f(np.mean(acc["n"], 0)[0]), cubes_per_bar=f(np.mean(acc["n"], 0)[1]), layers={})
    s1, s0, tok = np.array(acc["s1"]), np.array(acc["s0"]), np.array(acc["tok"])     # [pairs*3, L, 2], [pairs, L]
    for li in range(nL):
        o = dict(token_change=f(tok[:, li].mean()))
        for nm, s in (("oneform", s1), ("zeroform", s0)):
            if li >= s.shape[1]:
                continue
            x = s[:, li]
            x = x[np.isfinite(x).all(1)]
            o[nm] = dict(A=f(np.mean(x[:, 0]) / -S2), B=f(np.mean(x[:, 1]) / -S2),
                         A_over_B=f(np.mean(x[:, 0]) / np.mean(x[:, 1])),
                         split=f(np.mean(np.abs(x[:, 0] - x[:, 1])) / np.mean(np.abs(x[:, 1]))),
                         sign_agree=f(np.mean(np.sign(x[:, 0]) == np.sign(x[:, 1]))),
                         A_with=f(np.mean(x[:, 0] < 0)), B_with=f(np.mean(x[:, 1] < 0)), n=len(x))
        out["layers"][li] = o
    return out


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)                      # all-NaN cubes of partly hidden bars
    a = sys.argv[1:]
    model = a[0] if a and not a[0].startswith("--") else "dav2-s"
    nsc = int(a[a.index("--scenes") + 1]) if "--scenes" in a else None
    cap = int(a[a.index("--patches") + 1]) if "--patches" in a else None
    fd = os.path.join(ROOT, "outputs/feats", model)
    scenes = sorted(glob.glob(os.path.join(fd, "scene_*.npz")))[:nsc]
    evalf = sorted(f for f in glob.glob(os.path.join(fd, "*.npz")) if os.path.basename(f)[0] in "ABHR")
    d0 = np.load(scenes[0])
    nl, nf = d0["hs"].shape[0], d0["fus"].shape[0] if d0["fus"].ndim == 3 else 0
    names = [f"L{i}" for i in range(nl)] + [f"F{i + 1}" for i in range(nf)] + ["OUT"]
    nL = len(names)
    print(f"{model}: {len(scenes)} training scenes, {len(evalf)} stimuli/rings, layers {names[0]}..{names[-2]} + OUT")
    t0 = time.time()
    fits, groups = fit_probes(scenes, nl, nf, cap)
    exr = run_gain(groups[NFOLD], fits, nl, nf)
    print(f"probes fit, slope gain on {exr['0']['n_runs']} held-out runs ({time.time() - t0:.0f}s)", flush=True)

    R, meta = {}, {}
    for fn in evalf:
        n = os.path.basename(fn)[:-4]
        d = dict(np.load(fn))
        src = os.path.join(ROOT, "data/loops" if n[0] in "HR" else "data/stim2", n + ".npz")
        R[n] = evaluate(d, dict(np.load(src)), fits, nl, nf)
        if not re.fullmatch(r"(A_a\d+_p0_t0|B_a\d+_k\d)(_m)?", n):          # maps kept only for the pairs
            for key in ("P1", "P0", "cp"):
                del R[n][key]
        meta[n] = dict(set=str(d.get("set", n[0])), angle=int(d["angle"]), mirror=bool(d["mirror"]),
                       possible=int(d.get("possible", -1)))
    print(f"stimuli read out ({time.time() - t0:.0f}s)", flush=True)

    S = dict(model=model, n_scenes=len(scenes), patches_per_scene=cap or "all",
             scene_groups={str(g): len(v) for g, v in sorted(groups.items())},
             protocol=dict(alphas=[float(x) for x in ALPHAS], cv_folds=NFOLD, test="scenes with index % 5 == 0",
                           gcov_train=GCOV, fcov_train=FCOV, fcov_tokens=FBAR, cube_erosion=ERODE, dilation_patches=DIL,
                           calibration="raw / k, k = slope gain on held-out scene runs"),
             exact=dict(scene_runs=exr), layers={})
    for st in ("A", "B", "H", "R"):
        ns = [n for n in R if meta[n]["set"] == st]
        if ns:
            e1 = np.concatenate([R[n]["ex1"] / (R[n]["sg"] * S2) for n in ns])
            e0 = np.concatenate([R[n]["ex0"] / (R[n]["sg"] * S2) for n in ns])
            S["exact"][st] = dict(n=len(ns), oneform_ratio=f(np.nanmean(e1)), zeroform_ratio=f(np.nanmean(e0)),
                                  zeroform_ratio_sd=f(np.nanstd(e0)), nan_bars=int(np.sum(~np.isfinite(e0))),
                                  gamma=f(np.nanmedian([LT * np.sum(R[n]["ex1"]) for n in ns])),
                                  side_patches=f(np.mean([R[n]["nside"] for n in ns])),
                                  out_fullres_with=f(np.mean(np.concatenate([R[n]["out_full"] * R[n]["sg"]
                                                                             for n in ns]) > 0)),
                                  out_grid_vs_fullres_sign=f(np.mean(np.concatenate(
                                      [np.sign(R[n]["s0"][-1]) == np.sign(R[n]["out_full"]) for n in ns]))))
    for li, L in enumerate(names):
        e = {}
        for p, key, nm in (("1", "s1", "oneform"), ("0", "s0", "zeroform")):
            if (p, li) not in fits:
                continue
            ft = fits[p, li]
            e[nm] = {k: (v if k == "alpha" else f(v) if isinstance(v, float) else v) for k, v in ft.items()
                     if k not in ("A", "c")}
            e[nm].update(decodable=bool(ft["gain"] >= KMIN), **summarise(R, meta, key, li, ft["gain"]))
        S["layers"][L] = e
    ip = identical_pixels(R, fd, nl, nf, nL)
    if ip:
        for li, L in enumerate(names):
            S["layers"][L]["identical_pixels"] = ip["layers"][li]
        S["identical_pixels"] = {k: v for k, v in ip.items() if k != "layers"}
    print(f"done ({time.time() - t0:.0f}s)\n")

    pad = lambda x: np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 6 - x.shape[-1])], constant_values=np.nan)
    np.savez(os.path.join(ROOT, "outputs", f"layers_{model}.npz"), names=np.array(names), stims=np.array(list(R)),
             **{f"{q}1_{L}": fits["1", li][q] for li, L in enumerate(names[:-1]) for q in "Ac"},
             **{f"{q}0_{L}": fits["0", li][q] for li, L in enumerate(names) for q in "Ac"},
             s1=np.array([pad(R[n]["s1"]) for n in R]), s0=np.array([pad(R[n]["s0"]) for n in R]),
             sg=np.array([pad(R[n]["sg"]) for n in R]))
    json.dump(S, open(os.path.join(ROOT, "results", f"summary_layers_{model}.json"), "w"), indent=1)

    # ---------------------------------------------------------------- tables
    x = lambda v, fm="+6.2f": "   -  "[:len(format(0, fm))] if v is None else format(v, fm)
    for nm, title in (("oneform", "1-form probe (features -> depth gradient g)"),
                      ("zeroform", "0-form probe (features -> depth, centred per image; OUT = the network's output)")):
        print(f"{model} {title}\n  held-out scenes: R2, shrinkage s, slope gain k, share of runs with the right sign |"
              f" along-bar slope / exact -1/sqrt2 (raw; cal = raw / k);\n  with = share of bars running with the"
              f" objects (rings: with their exact sign); rho = A / controls of the same layout (median, 95% CI over"
              f" 15-deg bins);\n  Gamma = bars-only loop integral / T (cal; exact -1 on A and B, 0 on rings); AUC A vs"
              f" rings of |Gamma| and of -Gamma;\n  calibrated values and rho only where k >= {KMIN}")
        print("layer    R2     s     k  right |  A raw  A cal  with    rho  [95% CI]       |  B cal  with |"
              " rings  with | Gamma A      B  rings | AUC |G|  -G")
        for L in names:
            e = S["layers"][L].get(nm)
            if e is None:
                continue
            A, B, Rg = e.get("A", {}), e.get("B", {}), e.get("rings", {})
            ok = e["decodable"]
            rho, ci = (A.get("rho"), A.get("rho_ci", [None, None])) if ok else (None, [None, None])
            print(f"{L:5s} {x(e['r2'], '5.2f')} {x(e['shrink'], '5.2f')} {x(e.get('gain'), '5.2f')} "
                  f"{x(e.get('runs_right'), '5.2f')} | {x(A.get('ratio'))} {x(A.get('ratio_cal'))} "
                  f"{x(A.get('with_objects'), '5.2f')} {x(rho)} [{x(ci[0])},{x(ci[1])}] | "
                  f"{x(B.get('ratio_cal'))} {x(B.get('with_objects'), '5.2f')} | {x(Rg.get('ratio_cal'))} "
                  f"{x(Rg.get('with_objects'), '5.2f')} | {x(A.get('gamma'))} {x(B.get('gamma'))} "
                  f"{x(Rg.get('gamma'))} | {x(e.get('auc_gamma_A_vs_rings'), '5.2f')} "
                  f"{x(e.get('auc_signed_A_vs_rings'), '5.2f')}")
        print()
    if ip:
        print(f"{model} identical pixels: {ip['n_pairs']} pairs A_a*_p0_t0 / B_k; on the bars, read-outs that use no"
              f" patch within {DIL} patches of a differing pixel ({ip['side_patches_per_bar']:.1f} side-face patches,"
              f" {ip['cubes_per_bar']:.1f} cubes per bar)\n  slope / exact (raw), split = mean |A - B_k| / mean |B_k|,"
              f" agree = share of bars with the same sign; features = mean |F_A - F_B| / spread of F_B there")
        print("layer features | 1-form  A      B_k  split agree | 0-form  A      B_k  split agree")
        r = lambda q: f"{x(q['A'])} {x(q['B'])} {x(q['split'], '5.2f')} {x(q['sign_agree'], '5.2f')}" if q else " " * 26
        for li, L in enumerate(names):
            o = ip["layers"][li]
            print(f"{L:5s}  {x(o['token_change'], '6.3f')}  |  {r(o.get('oneform'))}  |  {r(o.get('zeroform'))}")
    print("\nexact targets through the same read-outs (1-form ratio, 0-form ratio, Gamma/T):",
          {k: (v["oneform_ratio"], v["zeroform_ratio"], v["gamma"]) for k, v in S["exact"].items()
           if k != "scene_runs"},
          "| scene runs, |exact slope| / (1/sqrt2):", {p: round(v["exact_abs_ratio"], 4) for p, v in exr.items()})
    print("network output, share of bars with the objects at full resolution (paper read-out) / sign agreement with the"
          " grid read-out:", {k: (v["out_fullres_with"], v["out_grid_vs_fullres_sign"]) for k, v in S["exact"].items()
                              if k != "scene_runs"})
