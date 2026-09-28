"""Check 6b: MiDaS (ssitrim + 0.5 GM) and E2E-FT (LS-aligned L1) on the full frame with a shared far wall
(regime B), evaluated exactly at the structured candidates: the three objects, the posterior mean, the
pointwise median (middle map), plus a short Adam refinement from the best candidate and from the mean.
usage: python check6b_background_losses.py <midas|lsl1> [gap_in_T=1.0] [adam_iters=150]"""
import sys, os, time, json
import numpy as np, torch
from common import load, split_of, mad_norm, wmedian3

torch.set_num_threads(int(os.environ.get("NT", "4")))
torch.set_default_dtype(torch.float64)
loss_name = sys.argv[1]
gapT = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 150
st = load("texture_a90"); fg = st["fg"]; H, W = fg.shape
T = 5 * np.sqrt(3)
obj = np.array([st["depth"][k] - np.median(st["depth"][k][fg]) for k in range(3)])
wall = np.nanmax(obj[:, fg]) + gapT * T
F = np.array([np.where(fg, obj[k], wall).ravel() for k in range(3)])
Mtot = F.shape[1]
Nn = np.array([mad_norm(F[k]) for k in range(3)])
Ft, Nt = torch.tensor(F), torch.tensor(Nn)
PS = [np.array(x) for x in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (1/3, 1/3, 1/3)]]


def N(f):
    md = f.median()
    return (f - md) / (f - md).abs().mean()


def gm(R):
    tot = 0
    for s in range(4):
        r = R[::2 ** s, ::2 ** s]
        tot = tot + ((r[:, 1:] - r[:, :-1]).abs().sum() + (r[1:, :] - r[:-1, :]).abs().sum()) / r.numel()
    return tot


def loss(f, p):
    tot = 0
    if loss_name == "lsl1":
        fc = f - f.mean()
        for k in range(3):
            dc = Ft[k] - Ft[k].mean()
            s = (fc @ dc) / (fc @ fc)
            tot = tot + p[k] * (s * fc - dc).abs().mean()
        return tot
    g = N(f)
    for k in range(3):
        r = g - Nt[k]
        keep = torch.topk(r.abs(), int(0.8 * Mtot), largest=False).values
        tot = tot + p[k] * (keep.sum() / (2 * Mtot) + 0.5 * gm(r.view(H, W)))
    return tot


def ev(v, p):
    with torch.no_grad():
        return float(loss(torch.tensor(v), p))


def fsplit(v):
    return split_of(np.where(fg, v.reshape(H, W), np.nan), st)


out = {}
for p in PS:
    t0 = time.time()
    base = F if loss_name == "lsl1" else Nn
    cands = {f"obj{k+1}": base[k] for k in range(3)}
    cands["mean"] = p @ base
    cands["pointwise median"] = wmedian3(base, p)
    vals = {k: ev(v, p) for k, v in cands.items()}
    best = min(vals, key=vals.get)
    line = f"[{loss_name}, wall gap {gapT}T] p={np.round(p, 3)}: " + ", ".join(f"{k}={v:.5f}" for k, v in vals.items())
    runs = {}
    for nm in dict.fromkeys([best, "mean"]):
        if iters <= 0:
            break
        f = torch.tensor(cands[nm].copy(), requires_grad=True)
        opt = torch.optim.Adam([f], lr=0.005 * float(np.abs(cands[nm]).mean() + 1e-9))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for it in range(iters):
            opt.zero_grad(); L = loss(f, p); L.backward(); opt.step(); sch.step()
        fn = f.detach().numpy()
        runs[nm] = (ev(fn, p), fsplit(fn).round(3).tolist())
    print(line + f" | best candidate {best}, split {np.round(fsplit(cands[best]), 3)} | Adam: {runs} ({time.time() - t0:.0f}s)", flush=True)
    out[str(np.round(p, 3).tolist())] = dict(candidates=vals, best=best, adam=runs)
json.dump(out, open(f"background_{loss_name}_gap{gapT}.json", "w"), indent=1)
