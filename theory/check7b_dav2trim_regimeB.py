"""Check 7b (docs/theory.md Sec. 3.5): DAv2 loss with 10% trimming of the ssi term in regime B (full 512x512 frame, the three
objects in the equal-median gauge in front of one shared wall at max object depth + gap; as check6_background.py).
Candidates: objects, mean, pointwise weighted median (object k if p_k > 1/2, else the middle map), then Adam
(300 steps, lr 0.01, as check6) from the mean and from the pointwise median.
usage: python check7b_dav2trim_regimeB.py [gapT=1.0] [iters=300] [trim=0.1]"""
import sys, os, time, json
import numpy as np, torch
from common import load, split_of, mad_norm, wmedian3

torch.set_num_threads(int(os.environ.get("NT", "4")))
torch.set_default_dtype(torch.float64)
gapT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300
trim = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
st = load("texture_a90"); fg = st["fg"]; H, W = fg.shape
T = 5 * np.sqrt(3); D = st["depth"]
obj = np.array([D[k] - np.median(D[k][fg]) for k in range(3)])
wall = np.nanmax(obj[:, fg]) + gapT * T
F = np.array([np.where(fg, obj[k], wall).ravel() for k in range(3)])
Mtot = F.shape[1]; KEEP = int((1 - trim) * Mtot)
Nn = np.array([mad_norm(F[k]) for k in range(3)]); Nt = torch.tensor(Nn)
PS = [np.array(x) for x in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (1/3, 1/3, 1/3), (0.45, 0.45, 0.1)]]


def N(f):
    md = f.median(); return (f - md) / (f - md).abs().mean()


def gm(R):
    tot = 0
    for s in range(4):
        r = R[::2 ** s, ::2 ** s]
        tot = tot + ((r[:, 1:] - r[:, :-1]).abs().sum() + (r[1:, :] - r[:-1, :]).abs().sum()) / r.numel()
    return tot


def loss(f, p):
    g = N(f); tot = 0
    for k in range(3):
        r = g - Nt[k]
        data = torch.topk(r.abs(), KEEP, largest=False).values.mean() if trim > 0 else r.abs().mean()
        tot = tot + p[k] * (data + 2 * gm(r.view(H, W)))
    return tot


def ev(v, p):
    with torch.no_grad():
        return float(loss(torch.tensor(v), p))


def fsplit(v):
    return split_of(np.where(fg, v.reshape(H, W), np.nan), st)


print(f"regime B, gap {gapT} T, ssi trimmed {trim:.0%} of the full frame ({trim / fg.mean():.0%} of the object's pixel count)")
out = {}
for p in PS:
    t0 = time.time()
    cands = {f"obj{k+1}": Nn[k] for k in range(3)}
    cands["mean"] = p @ Nn; cands["pointwise median"] = wmedian3(Nn, p)
    vals = {k: ev(v, p) for k, v in cands.items()}
    print(f"p={np.round(p,3)}: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()) +
          f" | split of pointwise median {np.round(fsplit(cands['pointwise median']),3)}", flush=True)
    runs = {}
    for nm in ["mean", "pointwise median"]:
        f = torch.tensor(cands[nm].copy(), requires_grad=True)
        opt = torch.optim.Adam([f], lr=0.01); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for it in range(iters):
            opt.zero_grad(); L = loss(f, p); L.backward(); opt.step(); sch.step()
        fn = f.detach().numpy()
        runs[nm] = dict(loss=ev(fn, p), split=np.round(fsplit(fn), 4).tolist())
        print(f"    Adam from {nm}: loss {runs[nm]['loss']:.4f}, split {runs[nm]['split']} ({time.time()-t0:.0f}s)", flush=True)
    out[str(np.round(p, 3).tolist())] = dict(candidates=vals, runs=runs)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"background_dav2trim_gap{gapT}.json"), "w"), indent=1)
