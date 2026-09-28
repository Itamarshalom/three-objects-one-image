"""Check 3 (docs/theory.md Sec. 3.1-3.2): L1 (median) predictors for the 3-object posterior, exact analysis.

(a) plain per-pixel L1 with fixed targets -> pointwise weighted median (two gauges)
(b) SSI-MAE with median/MAD normalisation of prediction AND target (MiDaS ssimae, Depth Anything L_ssi):
    L(f) = sum_k p_k mean|N(f) - N(d_k)|.  Relaxation min over all g gives a lower bound (pointwise median);
    it is attained iff the pointwise median is itself normalised (median 0, MAD 1).
    Otherwise: Lagrangian dual bound D(lam) = lam + mean_x min_v [c_x(v) - lam |v|] (v in {n_kx} U {0})
    and a primal patchwork g (one of the three normalised object values per pixel) -> certified bracket."""
import numpy as np
from common import load, split_of, to_img, mad_norm, wmedian3

st = load("texture_a90")
fg, cube, n = st["fg"], st["cube"][st["fg"]], st["n"]
X = st["depth"][:, fg]
Nn = np.array([mad_norm(X[k]) for k in range(3)])
M = X.shape[1]
PS = [(0.6, 0.3, 0.1), (0.8, 0.15, 0.05), (0.5, 0.3, 0.2), (0.45, 0.45, 0.1), (0.4, 0.35, 0.25), (0.36, 0.33, 0.31), (1/3, 1/3, 1/3)]


def fmt(v):
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def coeffs(f, basis):
    A = np.concatenate([basis.T, np.ones((M, 1))], 1)
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    r = A @ c - f
    return c[:3], np.sqrt((r ** 2).mean()) / f.std()


def ssimae(g, p):
    gn = mad_norm(g)
    return sum(p[k] * np.abs(gn - Nn[k]).mean() for k in range(3))


print("=== (a) pointwise weighted median, plain L1 with fixed targets ===")
for p in PS:
    raw = wmedian3(X, p)
    nrm = wmedian3(Nn, p)
    print(f"p={fmt(p)}  raw renderer gauge: split={fmt(split_of(to_img(raw, fg), st))} | "
          f"median/MAD gauge: split={fmt(split_of(to_img(nrm, fg), st))}, "
          f"MAD of median map={np.abs(nrm - np.median(nrm)).mean():.3f}, median={np.median(nrm):+.4f}")

# boundary p_1 = 1/2: every map pointwise between d_1 and the middle map is optimal (same loss)
p = np.array([0.5, 0.3, 0.2])
mid = np.sort(Nn, 0)[1]
for t in [0, 0.25, 0.5, 0.75, 1]:
    g = (1 - t) * Nn[0] + t * mid
    L = sum(p[k] * np.abs(g - Nn[k]).mean() for k in range(3))
    print(f"   p=(0.5,0.3,0.2), g=(1-t) n_1 + t*middle, t={t:.2f}: plain-L1 loss={L:.6f} split={fmt(split_of(to_img(g, fg), st))}")

print("\n=== (b) SSI-MAE (median/MAD on both), exact bracket ===")
lams = np.linspace(-0.5, 0.999, 3000)
for p in PS:
    p = np.array(p)
    cand = np.concatenate([Nn, np.zeros((1, M))])                       # candidate values per pixel [4, M]
    cost = np.array([sum(p[k] * np.abs(cand[j] - Nn[k]) for k in range(3)) for j in range(4)])
    absv = np.abs(cand)
    relax = sum(p[k] * np.abs(wmedian3(Nn, p) - Nn[k]).mean() for k in range(3))   # no-constraint bound
    D = np.array([l + (cost - l * absv).min(0).mean() for l in lams])
    lstar = lams[D.argmax()]; Dbest = D.max()
    # primal: pixelwise argmin at lam slightly above lam*, then top-up / trim to MAD = 1 by switching cost order
    lo = (cost - (lstar - 1e-6) * absv).argmin(0)
    hi = (cost - (lstar + 1e-3) * absv).argmin(0)
    g = cand[lo, np.arange(M)].copy()
    diff = np.where(lo != hi)[0]
    # switch pixels in order of the smallest extra cost per unit |v| gained, until mean|g| reaches 1
    gain = absv[hi[diff], diff] - absv[lo[diff], diff]
    extra = cost[hi[diff], diff] - cost[lo[diff], diff]
    order = diff[np.argsort(extra / np.maximum(gain, 1e-12))]
    need = M * 1.0 - np.abs(g).sum()
    for i in order:
        if need <= 0:
            break
        need -= absv[hi[i], i] - absv[lo[i], i]
        g[i] = cand[hi[i], i]
    Lp = ssimae(g, p)
    comp = [np.mean(np.isclose(g, Nn[k])) for k in range(3)]
    w, res = coeffs(mad_norm(g), Nn)
    best_obj = min(ssimae(X[k], p) for k in range(3))
    print(f"p={fmt(p)} relaxed bound={relax:.4f} | dual bound D(lam*={lstar:.3f})={Dbest:.4f} | primal patchwork loss={Lp:.4f} "
          f"(gap {Lp - Dbest:.4f}) | best single object={best_obj:.4f} | mean map={ssimae(p @ X, p):.4f} | "
          f"middle map={ssimae(np.sort(Nn, 0)[1], p):.4f}")
    print(f"      patchwork: pixel shares equal to obj1/2/3 = {np.round(comp, 3)}, split={fmt(split_of(to_img(g, fg), st))}, "
          f"span-fit resid={res:.3f}, median(g)={np.median(g):+.3f}")
    np.save(f"patch_{'_'.join(f'{x:.2f}' for x in p)}.npy", g)
