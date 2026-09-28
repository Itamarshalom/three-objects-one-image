"""Check 2: what an L2-trained predictor outputs for the 3-object posterior.

(a) plain L2 with a fixed target gauge            -> posterior mean, split = p
(b) per-target normalisation (Marigold / Lotus)   -> mean of normalised targets, split_k ~ p_k a_k
(c) MiDaS ssimse: least-squares scale+shift of the prediction onto each target, then MSE
    -> theory: top eigenvector of C = sum_k p_k d~_k d~_k^T, i.e. f = sum_k w_k d_k with P G w = lambda w
We optimise each population loss directly over the per-pixel prediction f (76k foreground pixels)."""
import numpy as np, torch
from common import load, split_of, to_img, pct_norm, mad_norm

torch.set_default_dtype(torch.float64)
st = load("texture_a90")
fg = st["fg"]
X = st["depth"][:, fg]                          # [3, M]
M = X.shape[1]
Xc = X - X.mean(1, keepdims=True)
G = Xc @ Xc.T / M
PS = [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (1/3, 1/3, 1/3), (0.5, 0.3, 0.2), (0.8, 0.15, 0.05), (0.45, 0.45, 0.1), (0.36, 0.33, 0.31)]


def coeffs(f):
    """least-squares f ~ sum_k w_k d_k + c on the foreground; returns w and relative residual"""
    A = np.concatenate([X.T, np.ones((M, 1))], 1)
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    r = A @ c - f
    return c[:3], np.sqrt((r ** 2).mean()) / f.std()


def fmt(v):
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def ssimse_loss(f, p, Xt):
    fc = f - f.mean()
    tot = 0
    for k in range(3):
        dk = Xt[k] - Xt[k].mean()
        s = (fc @ dk) / (fc @ fc)
        tot = tot + p[k] * ((s * fc - dk) ** 2).mean()
    return tot


def optimise(lossfn, f0, iters=300):
    f = torch.tensor(f0.copy(), requires_grad=True)
    opt = torch.optim.LBFGS([f], lr=1, max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-15,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L = lossfn(f)
        L.backward()
        return L
    opt.step(closure)
    return f.detach().numpy(), float(lossfn(f).detach())


def eig_theory(p):
    P = np.diag(p)
    lam, V = np.linalg.eig(P @ G)
    i = np.argmax(lam.real)
    w = V[:, i].real
    return w, lam.real


Xt = torch.tensor(X)
rng = np.random.default_rng(1)
print("Gram G (per pixel):\n", np.round(G, 4))
print("\n=== (a) plain L2, fixed gauge: optimum is the posterior mean (trivial); split(mean) ===")
for p in PS:
    print(f"p={fmt(p)} split(mean)={fmt(split_of(to_img(np.array(p) @ X, fg), st))}")

print("\n=== (b) per-target normalisation then plain L2: optimum = sum_k p_k N(d_k) ===")
for name, N in [("pct 2/98 -> [-1,1] (Marigold/Lotus)", pct_norm), ("median/MAD", mad_norm)]:
    Nn = np.array([N(X[k]) for k in range(3)])
    a = np.array([np.polyfit(X[k], Nn[k], 1)[0] for k in range(3)])
    print(f"{name}: scale a_k = {np.round(a, 5)}, a_k/mean(a) = {np.round(a / a.mean(), 5)}")
    for p in PS[:3]:
        s = split_of(to_img(np.array(p) @ Nn, fg), st)
        pa = np.array(p) * a / (np.array(p) * a).sum()
        print(f"   p={fmt(p)} split={fmt(s)}  predicted p_k a_k/sum={fmt(pa)}")

print("\n=== (c) MiDaS ssimse (LS scale+shift of prediction, then MSE) ===")
print("columns: p | theory w (top eig of P G), split=w/sum(w) | GD optimum split, sum(w)/|w|_1, resid | "
      "loss(opt) loss(mean) loss(best single obj) | p recovered as w/(G w)")
rows = []
for p in PS:
    p = np.array(p)
    wt, lam = eig_theory(p)
    wt = wt / np.abs(wt).sum() * np.sign(wt.sum() if abs(wt.sum()) > 1e-9 else 1)
    best = None
    for init in ["mean", "noise", "obj1", "obj3"]:
        f0 = {"mean": p @ X, "noise": rng.standard_normal(M), "obj1": X[0] + 0.1 * rng.standard_normal(M),
              "obj3": X[2] + 0.1 * rng.standard_normal(M)}[init]
        f, L = optimise(lambda f: ssimse_loss(f, p, Xt), f0)
        if best is None or L < best[1]:
            best = (f, L, init)
    f, L, init = best
    w, res = coeffs(f)
    w = w / np.abs(w).sum() * np.sign(w.sum() if abs(w.sum()) > 1e-9 else 1)
    Lmean = float(ssimse_loss(torch.tensor(p @ X), p, Xt))
    Lobj = min(float(ssimse_loss(Xt[k], p, Xt)) for k in range(3))
    sp = split_of(to_img(f, fg), st)
    prec = w / (G @ w); prec = prec / prec.sum()
    print(f"p={fmt(p)} | w_th={fmt(wt)} split_th={fmt(wt / wt.sum()) if abs(wt.sum()) > 1e-3 else 'undefined (sum w ~ 0)'} | "
          f"GD split={fmt(sp)} sum(w)/|w|1={w.sum():+.3f} resid={res:.1e} (best init {init}) | "
          f"L={L:.4f} L(mean)={Lmean:.4f} L(obj)={Lobj:.4f} | p_rec={fmt(prec)}")
    rows.append((p, wt, sp, L, Lmean, Lobj))
np.save("ssimse_results.npy", np.array([np.concatenate([r[0], r[1], r[2], [r[3], r[4], r[5]]]) for r in rows]))

# closed form with the circulant approximation G = c I + gamma J: split_j = gamma p_j / (lambda - c p_j)
c = np.diag(G).mean() - G[np.triu_indices(3, 1)].mean(); g = G[np.triu_indices(3, 1)].mean()
from scipy.optimize import brentq
print(f"\ncirculant closed form (c={c:.3f}, gamma={g:.3f}):")
for p in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (0.8, 0.15, 0.05), (0.36, 0.33, 0.31)]:
    p = np.array(p); ps = np.sort(p)[::-1]
    h = lambda lam: (p / (lam - c * p)).sum() - 1 / g
    lam = brentq(h, c * ps[1] + 1e-9, c * ps[0] - 1e-9)
    print(f"  p={fmt(p)} lambda={lam:.4f} split = gamma p/(lambda - c p) = {fmt(g * p / (lam - c * p))}")

# (d) median/MAD normalisation of BOTH prediction and target, then MSE (variant)
print("\n=== (d) variant: N(f) and N(d_k) both median/MAD normalised, then MSE ===")
Nn = torch.tensor(np.array([mad_norm(X[k]) for k in range(3)]))


def madmse(f, p):
    m = f.median()
    g = (f - m) / (f - m).abs().mean()
    return sum(p[k] * ((g - Nn[k]) ** 2).mean() for k in range(3))


for p in [(0.6, 0.3, 0.1), (1/3, 1/3, 1/3)]:
    f = torch.tensor(np.array(p) @ X, requires_grad=True)
    opt = torch.optim.Adam([f], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    for it in range(3000):
        opt.zero_grad(); L = madmse(f, p); L.backward(); opt.step(); sched.step()
    fn = f.detach().numpy()
    w, res = coeffs(fn)
    print(f"p={fmt(p)} loss={float(madmse(f.detach(), p)):.4f} (mean map: {float(madmse(torch.tensor(np.array(p) @ X), p)):.4f}) "
          f"split={fmt(split_of(to_img(fn, fg), st))} resid of span fit={res:.3f}")
