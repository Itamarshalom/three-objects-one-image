"""Check 14 (docs/theory.md Sec. 7): the paper's read-out on every optimum and candidate map the theory finds.

For each map we take the paper's loop profile (per-cube median over pixels >= 3 px inside the cube, pdepth.readout),
the bar slopes b_j and joint jumps J_j of pdepth.loop.bars_and_joints, and the excess tear of
scripts/analyze_excess.py with ground-truth references instead of a network's controls:
    e_j = (J_j - r_j) / (g_j - r_j),   r_j = mean step at joint j of the two objects whose gap is elsewhere,
                                        g_j = step at joint j of the object whose gap is there.
beta_j = b_j / (the objects' common bar slope): 1 = recedes like an object, 0 = flat, -1 = reversed.
Two scales. 'native': the map as built (weights summing to one for maps made of the objects). 'MAD': the map
median/MAD-normalised to the objects' MAD, where a median/MAD-normalised loss puts it; the scale-invariant losses
leave the output scale free, so this is one choice among many. A shift never matters.
Maps: the objects; the mean at several p; the middle map (pointwise median of the median-centred objects); the
least-squares-aligned L2 optimum (top eigenvector of diag(p) G, check2); the best map found for each (loss, p) by
check3b (opt/*.json) and check7 (opt7/*.json); the SSI-MAE patchworks of check3; means over the objects and their
depth reversals, w * mean(p) - (1 - w) * mean(q); and (8) what the same algebra gives in a network's own units, where
the ordinary-joint steps r_j are not zero (from the possible controls in results/).
usage: python check14_excess.py > check14_excess.log   (needs opt/*.json and opt7/*.json: python run_opt.py)"""
import os, sys, json, glob
import numpy as np
from common import load, mad_norm, paper_profile, ensure, ROOT

sys.path.insert(0, ROOT)
from pdepth.loop import bars_and_joints  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
st = load("texture_a90"); fg = st["fg"]
X = st["depth"][:, fg]
MAD0 = np.mean([np.mean(np.abs(x - np.median(x))) for x in X])          # the objects' MAD (they agree to 0.2%)

# ---- ground-truth references
BJ = [bars_and_joints(paper_profile(X[k], st)) for k in range(3)]
B_OBJ = np.mean([b for b, _ in BJ])
real = np.array([np.mean([BJ[k][1][j] for k in range(3) if k != j]) for j in range(3)])
tear = np.array([BJ[j][1][j] for j in range(3)])
print(f"objects: bar slopes {np.round([b for b, _ in BJ], 4).tolist()} (common slope {B_OBJ:.4f} per cube step)")
print(f"references: real-joint step r = {np.round(real, 4)}, gap step g = {np.round(tear, 4)}, "
      f"T = 5 sqrt3 = {5 * np.sqrt(3):.4f}; objects' MAD {MAD0:.4f}")


def read(y):
    """e and beta of a loop profile y (object depth units)"""
    b, J = bars_and_joints(np.asarray(y, float))
    return (J - real) / (tear - real), b / B_OBJ


def row(name, native=None, mad=None, extra=""):
    """print e and beta at the native scale (foreground vector) and at MAD scale (foreground vector or profile)"""
    for lab, v in (("native", native), ("MAD", mad)):
        if v is None:
            continue
        y = paper_profile(v, st) if len(v) == len(X[0]) else np.asarray(v)
        e, beta = read(y)
        print(f"  {name:44s} {lab:6s} e = ({e[0]:+.3f}, {e[1]:+.3f}, {e[2]:+.3f})  sum e = {e.sum():+.3f} | "
              f"beta = ({beta[0]:+.2f}, {beta[1]:+.2f}, {beta[2]:+.2f}) {extra}")
        name, extra = "", ""


def to_mad(v):
    return mad_norm(v) * MAD0


# the read-out's own invariance: a shift of each output and a scale shared by an image and its references
m = np.array([0.5, 0.3, 0.2]) @ X
e0, _ = read(paper_profile(m, st))
a, c = -2.5, 7.0
BJa = [bars_and_joints(paper_profile(a * X[k] + c + k, st)) for k in range(3)]
ra = np.array([np.mean([BJa[k][1][j] for k in range(3) if k != j]) for j in range(3)]); ga = np.array([BJa[j][1][j] for j in range(3)])
ea = (bars_and_joints(paper_profile(a * m - 3.0, st))[1] - ra) / (ga - ra)
print(f"invariance: e of the mean at p=(0.5,0.3,0.2) {np.round(e0, 4)}; with map and references scaled by {a} and "
      f"shifted separately {np.round(ea, 4)}; map alone scaled by 2 {np.round(read(paper_profile(2 * m, st))[0], 4)}")

print("\n(1) the three objects")
for k in range(3):
    row(f"object {k + 1}", X[k], to_mad(X[k]))

print("\n(2) the mean sum_k p_k d_k (squared error, fixed targets: Prop. 1)")
for p in [(0.6, 0.3, 0.1), (0.5, 0.5, 0.0), (0.45, 0.45, 0.1), (0.4, 0.35, 0.25), (1 / 3, 1 / 3, 1 / 3)]:
    m = np.array(p) @ X
    row(f"mean, p = {np.round(p, 2).tolist()}", m, to_mad(m))

print("\n(3) the middle map (per-pixel absolute error, no majority: Prop. 5, Lemma 6)")
mid = np.sort(X - np.median(X, 1, keepdims=True), 0)[1]
row("middle of the median-centred objects", mid, to_mad(mid))
row("middle of the objects as rendered (= object 2)", np.sort(X, 0)[1])

print("\n(4) least-squares-aligned L2 optimum: top eigenvector w of diag(p) G (check2; scale and sign free)")
Xc = X - X.mean(1, keepdims=True); G = Xc @ Xc.T / X.shape[1]
for p in [(0.6, 0.3, 0.1), (0.45, 0.45, 0.1), (0.4, 0.35, 0.25), (1 / 3, 1 / 3, 1 / 3)]:
    ev, V = np.linalg.eig(np.diag(p) @ G); w = np.real(V[:, np.argmax(np.real(ev))])
    w = w * (np.sign(w.sum()) if abs(w.sum()) > 1e-6 * np.abs(w).sum() else np.sign(w[np.argmax(p)]))
    row(f"LS-L2 optimum, p = {np.round(p, 2).tolist()}", mad=to_mad(w @ Xc),
        extra=f"w/sum|w| = {np.round(w / np.abs(w).sum(), 3).tolist()}")

print("\n(5) check3b / check7 (object only, regime A): every Adam end, read at MAD scale. * = beats every object, so the")
print("    best map found is not an object; otherwise the best map found is the best object (sum e = 1, beta = 1)")
files = sorted(glob.glob(os.path.join(HERE, "opt", "*.json"))) + sorted(glob.glob(os.path.join(HERE, "opt7", "*.json")))
if not files:
    print("  no opt/*.json or opt7/*.json: run python run_opt.py first")
for f in files:
    r = json.load(open(f))
    objs = {k: v for k, v in r["candidates"].items() if k.startswith("obj")}
    kobj = min(objs, key=objs.get)
    print(f"  {r['loss']} p={np.round(r['p'], 2).tolist()}: best object {kobj} {objs[kobj]:.4f}")
    for kb, run in r["runs"].items():
        if "profile" in run:
            y = np.array(run["profile"]) * MAD0
        else:
            npy = f[:-5] + f"_{kb}.npy"
            if not os.path.exists(npy):
                print(f"    Adam from {kb}: no stored profile and no {os.path.basename(npy)}; rerun run_opt.py")
                continue
            y = paper_profile(to_mad(np.load(npy)), st)
        star = "*" if run["loss"] < objs[kobj] - 1e-4 else " "
        row(f"  Adam from {kb}", mad=y, extra=f"loss {run['loss']:.4f}{star}")

print("\n(6) SSI-MAE (median/MAD absolute error) patchworks of check3, read at MAD scale")
TAGS = ["0.50_0.30_0.20", "0.45_0.45_0.10", "0.40_0.35_0.25", "0.36_0.33_0.31", "0.33_0.33_0.33"]
ensure([f"patch_{t}.npy" for t in TAGS], ["check3_l1.py"])
for t in TAGS:
    row(f"SSI-MAE patchwork, p = ({t.replace('_', ', ')})", mad=to_mad(np.load(os.path.join(HERE, f"patch_{t}.npy"))))

print("\n(7) mean over the objects (weight w, belief p) and their depth reversals (weight 1 - w, belief q)")
u = np.full(3, 1 / 3)
for q, lab in [(u, "q = p"), (np.array([1.0, 0, 0]), "q = object 1")]:
    for w in [1.0, 0.75, 0.6, 0.5, 0.4, 0.25, 0.0]:
        m = w * (u @ X) - (1 - w) * (q @ X)
        row(f"p uniform, {lab}, w = P(upright) = {w:.2f}", m, extra=f"(P(up) - P(rev) = {2 * w - 1:+.2f})")

# (8) in a network's own units the ordinary-joint steps r_j are not zero. For a combination sum_k c_k P_k of the loop
# profiles P_k of the network's own controls, e_j = c_j + (C - 1) rho_j with C = sum_k c_k and rho_j = r_j / (g_j - r_j)
# (exact when the two controls whose gap is not at j get equal weights, as for the negated average). A profile with no
# jump at any joint has e_j = -rho_j; the negated average of the controls (C = -1) has sum e = -1 - 2 sum_j rho_j.
print("\n(8) network units, from each network's possible controls (results/, set B, seed 0): median over layouts")
RES = os.path.join(ROOT, "results")
runs = {m: f"readout_{m}.csv" for m in ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft"]}
runs.update({"mg11-1": "mg11_1step.csv", "mg10-1": "mg10_1step.csv"})
if not os.path.isdir(RES):
    print("  results/ not found")
else:
    from pdepth.analysis import load as load_csv
    PR = [f"prof{j}" for j in range(15)]
    for m, fn in runs.items():
        if not os.path.exists(os.path.join(RES, fn)):
            continue
        ctrl = {}
        for r in load_csv(os.path.join(RES, fn)):
            if r["set"] == "B" and r.get("seed", 0) in (0, "", None):
                ctrl.setdefault((r["angle"], r["mirror"]), {})[int(r["possible"])] = \
                    bars_and_joints(np.array([r[q] for q in PR], float))[1]
        cont, neg, rg = [], [], []
        for d in ctrl.values():
            if len(d) < 3:
                continue
            rn = np.array([np.mean([d[k][j] for k in range(3) if k != j]) for j in range(3)])
            gn = np.array([d[j][j] for j in range(3)])
            rho = rn / (gn - rn)
            cont.append(-rho.sum()); neg.append(-1 - 2 * rho.sum()); rg.extend(rn / gn)
        print(f"  {m:7s} ({len(cont)} layouts): ordinary step r_j / g_j {np.median(rg):+.2f} | sum e of a map with no jump "
              f"at any joint {np.median(cont):+.2f} | of the negated average of its controls {np.median(neg):+.2f}")
