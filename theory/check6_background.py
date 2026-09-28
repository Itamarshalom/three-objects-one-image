"""Check 6: the same losses when the loss sees the whole frame and the three objects share a far back wall.

Real nets are trained on full images, so the background is part of the target.  Here every object is placed
with the same foreground median depth (equal-median gauge) in front of one wall at depth (max object depth + gap);
the wall is identical in all three targets.  We redo:
  (i)   LS-aligned SSI-L2 (MiDaS ssimse): eigen-solution (exact, validated by GD in check2)
  (ii)  SSI-MAE (median/MAD): is the pointwise weighted median itself normalised?  If yes it is the exact optimum.
  (iii) Depth Anything V2 loss (SSI-MAE + 2 GM): exact loss of the structured candidates (objects, mean, pointwise
        median, all 729 half-bar patchworks), then Adam from the mean and from the pointwise median.
usage: python check6_background.py [gap_in_T_units=1.0] [adam_iters=300] [gauge=median|raw] [skip_iii=0|1]"""
import sys, os, itertools, time, json
import numpy as np, torch
from common import load, split_of, mad_norm, wmedian3

torch.set_num_threads(int(os.environ.get("NT", "4")))
torch.set_default_dtype(torch.float64)
gapT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300
gauge = sys.argv[3] if len(sys.argv) > 3 else "median"
skip3 = len(sys.argv) > 4 and sys.argv[4] == "1"
st = load("texture_a90"); fg = st["fg"]; H, W = fg.shape
T = 5 * np.sqrt(3)
D = st["depth"]
obj = np.array([D[k] - np.median(D[k][fg]) for k in range(3)]) if gauge == "median" else D.copy()
wall = np.nanmax(obj[:, fg]) + gapT * T
F = np.array([np.where(fg, obj[k], wall).ravel() for k in range(3)])     # [3, H*W]
Mtot = F.shape[1]
print(f"gauge = {gauge}; gap = {gapT} T, wall at {wall:.2f}; foreground share {fg.mean():.3f}; object depth range {np.ptp(obj[:, fg]):.2f}")
PS = [np.array(x) for x in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (1/3, 1/3, 1/3), (0.45, 0.45, 0.1)]]


def fsplit(v):
    return split_of(np.where(fg, v.reshape(H, W), np.nan), st)


def fmt(v):
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


# (i) LS-aligned SSI-L2
Fc = F - F.mean(1, keepdims=True)
G = Fc @ Fc.T / Mtot
c = np.diag(G).mean() - G[np.triu_indices(3, 1)].mean(); g = G[np.triu_indices(3, 1)].mean()
print(f"(i) full-frame Gram: c = {c:.3f}, gamma = {g:+.3f} (gamma > 0 means a shared component dominates)")
for p in PS:
    lam, V = np.linalg.eig(np.diag(p) @ G)
    w = V[:, np.argmax(lam.real)].real
    print(f"    p={fmt(p)}  LS-SSI-L2 optimum split = {fmt(w / w.sum()) if abs(w.sum()) > 1e-6 * np.abs(w).sum() else 'undefined'}")

# (ii) SSI-MAE: pointwise weighted median of the normalised targets
Nn = np.array([mad_norm(F[k]) for k in range(3)])
print("(ii) normalised targets: median/MAD of raw full maps:", np.round([np.median(F[k]) for k in range(3)], 3),
      np.round([np.abs(F[k] - np.median(F[k])).mean() for k in range(3)], 4))
for p in PS:
    m = wmedian3(Nn, p)
    feas_med, feas_mad = np.median(m), np.abs(m - np.median(m)).mean()
    L_m = sum(p[k] * np.abs(mad_norm(m) - Nn[k]).mean() for k in range(3))
    L_bound = sum(p[k] * np.abs(m - Nn[k]).mean() for k in range(3))
    print(f"    p={fmt(p)} pointwise median: median={feas_med:+.4f} MAD={feas_mad:.4f} -> loss after normalising {L_m:.5f} "
          f"vs relaxed bound {L_bound:.5f}; split={fmt(fsplit(m))}")

# (iii) Depth Anything V2 loss on the full frame
if skip3:
    sys.exit(0)
Nt = torch.tensor(Nn)


def N(f):
    md = f.median()
    return (f - md) / (f - md).abs().mean()


def gm(R):
    tot = 0
    for s in range(4):
        r = R[::2 ** s, ::2 ** s]
        tot = tot + ((r[:, 1:] - r[:, :-1]).abs().sum() + (r[1:, :] - r[:-1, :]).abs().sum()) / r.numel()
    return tot


def loss(f, p, gm_w=2.0):
    gg = N(f); tot = 0
    for k in range(3):
        r = gg - Nt[k]
        tot = tot + p[k] * (r.abs().mean() + gm_w * gm(r.view(H, W)))
    return tot


def ev(v, p):
    with torch.no_grad():
        return float(loss(torch.tensor(v), p))


mid_fg = np.sort(Nn, 0)[1]
barid = np.full(Mtot, -1)
barid[fg.ravel()] = np.minimum(st["cube"][fg] // (st["n"] // 3), 2)
half = np.where(barid >= 0, barid * 2 + (mid_fg < np.median(mid_fg[barid >= 0])), -1)
out = {}
for p in PS:
    t0 = time.time()
    cands = {f"obj{k+1}": Nn[k] for k in range(3)}
    cands["mean"] = p @ Nn
    cands["pointwise median"] = wmedian3(Nn, p)
    vals = {k: ev(v, p) for k, v in cands.items()}
    hb = []
    for combo in itertools.product(range(3), repeat=6):
        v = Nn[0].copy()
        sel = half >= 0
        v[sel] = Nn[np.array(combo)[half[sel]], np.flatnonzero(sel)]
        hb.append((ev(v, p), combo))
    hb.sort()
    print(f"(iii) p={fmt(p)} DAv2 loss of candidates: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()) +
          f" | best half-bar maps {[(round(L, 4), c_) for L, c_ in hb[:3]]} ({time.time() - t0:.0f}s)", flush=True)
    runs = {}
    starts = {"mean": cands["mean"], "pointwise median": cands["pointwise median"]}
    for nm, v0 in starts.items():
        if iters <= 0:
            break
        f = torch.tensor(v0.copy(), requires_grad=True)
        opt = torch.optim.Adam([f], lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for it in range(iters):
            opt.zero_grad(); L = loss(f, p); L.backward(); opt.step(); sch.step()
        fn = f.detach().numpy()
        runs[nm] = dict(loss=ev(fn, p), split=fsplit(fn).round(4).tolist())
        print(f"      Adam from {nm}: loss {runs[nm]['loss']:.4f}, split {fmt(fsplit(fn))} ({time.time() - t0:.0f}s)", flush=True)
    out[str(p.round(3).tolist())] = dict(candidates=vals, halfbar_top3=[(L, list(c_)) for L, c_ in hb[:3]], runs=runs)
json.dump(out, open(f"background_{gauge}_gap{gapT}.json", "w"), indent=1)
