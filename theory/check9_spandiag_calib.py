"""Check 9 (docs/theory.md Sec. 1 and 6): does the span diagnostic (split w/sum w, ramp share sum w/sum|w|, profile residual) separate
the theory's cases (object / posterior mean / SSI-L2 principal component / SSI-MAE patchwork) at the error level of
real outputs?  And does a statistic aimed at the patchwork (in-bar steps off the ramp) do better?

Theory maps (15-cube profiles on texture_a90): objects 1-3; posterior means for 3 p; SSI-L2 PCA optima for 2 p
(Theorem 3); SSI-MAE Lagrangian patchworks for 5 p (check3 patch_*.npy).
Perturbations: (a) iid per-cube noise scaled so the relative profile residual of an in-span map is rho;
(b) the per-cube residual pattern of DAv2 S/B/L's own output on the one-answer control (random sign, random
    reflection of the loop is NOT applied: the pattern is tied to the image), scaled to the same relative size.
Classifier: nearest clean template in (split, ramp share, residual) space -- a best case for the diagnostic.
Patchwork statistic: z_i = (in-bar step_i - ramp) / std(profile) over the 12 in-bar steps; report max|z|."""
import os
import numpy as np
from common import load, cube_profile, pilot_profiles, ensure, OUT

st = load("texture_a90"); fg = st["fg"]; n = st["n"]
P = np.array([cube_profile(st["depth"][k], st["cube"], n) for k in range(3)])
A = np.concatenate([P.T, np.ones((n, 1))], 1)
Q, _ = np.linalg.qr(A)
corners = np.asarray(st["corners"])
into = (corners - 1) % n
inbar = np.setdiff1d(np.arange(n), into)


def diag(y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None); w = c[:3]
    res = np.linalg.norm(A @ c - y) / np.linalg.norm(y - y.mean())
    return np.r_[w / w.sum(), w.sum() / np.abs(w).sum(), res]


def inbar_z(y):
    s = np.roll(y, -1) - y
    ramp = np.median(s[inbar])
    return np.abs(s[inbar] - ramp) / y.std()


# ---- theory maps
X = st["depth"][:, fg]
Xc = X - X.mean(1, keepdims=True); G = Xc @ Xc.T / X.shape[1]
maps = {}
for k in range(3):
    maps[f"object{k+1}"] = ("object", P[k])
for p in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (1/3, 1/3, 1/3)]:
    maps[f"mean{np.round(p,2)}"] = ("mean", np.array(p) @ P)
for p in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25)]:
    ev, V = np.linalg.eig(np.diag(p) @ G); w = np.real(V[:, np.argmax(np.real(ev))])
    w = w * np.sign(w.sum())
    maps[f"PCA{np.round(p,2)}"] = ("PCA", w @ P)
TAGS = ["0.40_0.35_0.25", "0.36_0.33_0.31", "0.45_0.45_0.10", "0.50_0.30_0.20", "0.33_0.33_0.33"]
ensure([f"patch_{t}.npy" for t in TAGS], ["check3_l1.py"])         # patchworks written by check3_l1.py
for tag in TAGS:
    g = np.load(os.path.join(OUT, f"patch_{tag}.npy"))
    im = np.full(fg.shape, np.nan); im[fg] = g
    maps[f"patch{tag}"] = ("patchwork", cube_profile(im, st["cube"], n))

print("clean theory maps: split | ramp share | residual | max in-bar |z|")
tmpl = {}
for nm, (cls, y) in maps.items():
    d = diag(y); tmpl[nm] = (cls, d)
    print(f"  {nm:24s} {cls:9s} split {np.round(d[:3],2)} ramp {d[3]:+.2f} resid {d[4]:.2f} | max|z| {inbar_z(y).max():.2f}")

# ---- pilot outputs: control residual patterns and statistics
print("\npilot outputs (texture_a90 and its one-answer control):")
ctrl = {}
for nm, y in pilot_profiles("*texture_a90*.npy"):          # bundled profiles if the pilot maps are absent
    if not (nm.startswith("dav2") or "s1_" in nm or "s25_" in nm): continue
    d = diag(y); z = inbar_z(y)
    print(f"  {nm:40s} split {np.round(d[:3],2)} ramp {d[3]:+.2f} resid {d[4]:.2f} | max in-bar |z| {z.max():.2f}, "
          f"median {np.median(z):.2f}")
    if nm.startswith("dav2") and "possible0" in nm:
        c, *_ = np.linalg.lstsq(A, y, rcond=None); r = y - A @ c
        ctrl[nm.split("_")[0]] = (r / np.linalg.norm(y - y.mean()), z)

# ---- confusion under perturbations
classes = ["object", "mean", "PCA", "patchwork"]
names = list(tmpl)
TM = np.array([tmpl[k][1] for k in names])


def classify(d):
    return tmpl[names[np.argmin(((TM - d) ** 2).sum(1))]][0]


rng = np.random.default_rng(0)
print("\nconfusion (rows = true class, cols = nearest-template label; share of 400 draws per map, pooled per class)")
for kind, rho in [("iid", 0.05), ("iid", 0.1), ("iid", 0.2), ("iid", 0.3)] + [("ctrl-" + m, None) for m in ctrl]:
    conf = {c: np.zeros(4) for c in classes}
    zmax = {c: [] for c in classes}
    for nm, (cls, y) in maps.items():
        sc = np.linalg.norm(y - y.mean())
        for _ in range(400):
            if kind == "iid":
                # iid per-cube noise, sd chosen so that its out-of-span part has expected norm rho * |y - mean y|
                yp = y + rho * sc / np.sqrt(n - 4) * rng.standard_normal(n)
            else:
                r, _ = ctrl[kind[5:]]
                yp = y + rng.choice([-1, 1]) * r * sc * rng.uniform(0.8, 1.2)
            d = diag(yp)
            conf[cls][classes.index(classify(d))] += 1
            zmax[cls].append(inbar_z(yp).max())
    lab = kind if rho is None else f"{kind} rho={rho}"
    print(f"  [{lab}]")
    for c in classes:
        row = conf[c] / conf[c].sum()
        print(f"    {c:9s} -> " + "  ".join(f"{cc}:{v:.2f}" for cc, v in zip(classes, row)) +
              f" | max in-bar |z|: median {np.median(zmax[c]):.2f} (10-90% {np.percentile(zmax[c],10):.2f}-{np.percentile(zmax[c],90):.2f})")

# ---- (c) full control error: the control output, affinely aligned to object 1, minus object 1 (in-span bias such as
#          a flattened ramp included), added to every theory map at the same relative size; plus the patchwork
#          statistic thresholded at 1.25 x the control's own max in-bar |z|
print("\n(c) full control error (aligned control output - object 1) added to each theory map")
for nm, yc in pilot_profiles("dav2*_texture_a90_possible0.npy"):
    m = nm.split("_")[0]
    B = np.stack([yc, np.ones(n)], 1); cc, *_ = np.linalg.lstsq(B, P[0], rcond=None)
    err = (B @ cc - P[0]) / np.linalg.norm(P[0] - P[0].mean())
    zc = inbar_z(yc).max()
    conf = {c: np.zeros(4) for c in classes}; over = {c: [] for c in classes}; ramps = {c: [] for c in classes}
    for nm, (cls, y) in maps.items():
        sc = np.linalg.norm(y - y.mean())
        for _ in range(200):
            yp = y + err * sc * rng.uniform(0.8, 1.2) + 0.05 * sc / np.sqrt(n - 4) * rng.standard_normal(n)
            d = diag(yp)
            conf[cls][classes.index(classify(d))] += 1
            over[cls].append(inbar_z(yp).max() > 1.25 * zc); ramps[cls].append(d[3])
    print(f"  [{m}] relative size of the control error {np.linalg.norm(err):.2f}; control max in-bar |z| {zc:.2f}")
    for c in classes:
        row = conf[c] / conf[c].sum()
        print(f"    {c:9s} -> " + "  ".join(f"{cc_}:{v:.2f}" for cc_, v in zip(classes, row)) +
              f" | ramp share median {np.median(ramps[c]):+.2f} | share with max in-bar |z| > 1.25 x control: {np.mean(over[c]):.2f}")
