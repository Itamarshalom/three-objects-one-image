"""Check 3b (docs/theory.md Sec. 3.3-3.4): population optimum of the robust SSI losses used by real depth nets, by direct optimisation
over the per-pixel prediction f (Adam from several initialisations; the loss is non-convex).

losses (per target k, then averaged with weights p):
  ssimae  : mean|N(f) - N(d_k)|                                  (Depth Anything V1 L_ssi; MiDaS ssimae)
  dav2    : ssimae + 2 * GM(N(f) - N(d_k))                        (Depth Anything V2: L_ssi : L_gm = 1 : 2)
  midas   : (1/2M) sum of the 80% smallest |N(f)-N(d_k)| + 0.5 * GM  (MiDaS ssitrim + alpha L_reg)
  lsl1    : mean|LS-aligned f - d_k|                              (Martin Garcia et al. E2E-FT depth loss)
GM = MiDaS multi-scale gradient matching, 4 scales, sum |grad_x R| + |grad_y R| over valid pairs / #valid pixels.
N(x) = (x - median x) / mean|x - median x|.
usage: python check3b_opt.py <loss> <p1,p2,p3> [iters]"""
import sys, os, time, json
import numpy as np, torch
from common import load, split_of, to_img, mad_norm, paper_profile

torch.set_num_threads(int(os.environ.get("NT", "4")))
torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "opt"), exist_ok=True)
loss_name = sys.argv[1]
p = np.array([float(x) for x in sys.argv[2].split(",")]); p = p / p.sum()
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 400
if loss_name == "ssimae" and os.environ.get("RUN_SSIMAE") != "1":
    sys.exit("ssimae has the exact Lagrangian bracket in check3_l1.py; set RUN_SSIMAE=1 to optimise anyway")

st = load("texture_a90")
fg = st["fg"]; H, W = fg.shape
X = st["depth"][:, fg]; M = X.shape[1]
Nn = np.array([mad_norm(X[k]) for k in range(3)])
idx = torch.tensor(np.flatnonzero(fg.ravel()))
mask = torch.tensor(fg, dtype=torch.float64)
Xt, Nt = torch.tensor(X), torch.tensor(Nn)


def N(f):
    m = f.median()
    return (f - m) / (f - m).abs().mean()


def img(v):
    out = torch.zeros(H * W)
    out = out.index_put((idx,), v)
    return out.view(H, W)


def gm(R):
    """MiDaS gradient loss on residual image R (zero outside mask), 4 scales, batch-based reduction"""
    tot = 0
    for s in range(4):
        st_ = 2 ** s
        r, mk = R[::st_, ::st_], mask[::st_, ::st_]
        gx = (r[:, 1:] - r[:, :-1]).abs() * mk[:, 1:] * mk[:, :-1]
        gy = (r[1:, :] - r[:-1, :]).abs() * mk[1:, :] * mk[:-1, :]
        tot = tot + (gx.sum() + gy.sum()) / mk.sum()
    return tot


def loss(f):
    if loss_name == "lsl1":
        fc = f - f.mean(); tot = 0
        for k in range(3):
            dk = Xt[k]; dc = dk - dk.mean()
            s = (fc @ dc) / (fc @ fc)
            tot = tot + p[k] * (s * fc + dk.mean() - dk).abs().mean()
        return tot
    g = N(f); tot = 0
    for k in range(3):
        r = g - Nt[k]
        if loss_name == "midas":
            a = r.abs()
            keep = torch.topk(a, int(0.8 * M), largest=False).values
            data = keep.sum() / (2 * M)
        else:
            data = r.abs().mean()
        tot = tot + p[k] * data
        if loss_name == "dav2":
            tot = tot + p[k] * 2 * gm(img(r))
        elif loss_name == "midas":
            tot = tot + p[k] * 0.5 * gm(img(r))
    return tot


def coeffs(f):
    A = np.concatenate([X.T, np.ones((M, 1))], 1)
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    r = A @ c - f
    return c[:3], np.sqrt((r ** 2).mean()) / f.std()


def evaluate(f):
    with torch.no_grad():
        return float(loss(torch.tensor(f)))


rng = np.random.default_rng(0)
tag = "_".join(f"{x:.2f}" for x in p)
# structured candidates: on each of the 6 half-bars (bar x sign of the middle map) take one object's normalised value
mid = np.sort(Nn, 0)[1]
bar = np.minimum(st["cube"][fg] // (st["n"] // 3), 2)
half = bar * 2 + (mid < 0)
import itertools
t0 = time.time()
cands = []
for combo in itertools.product(range(3), repeat=6):
    g = Nn[np.array(combo)[half], np.arange(M)]
    cands.append((evaluate(g), combo))
cands.sort()
print(f"half-bar search ({len(cands)} maps, {time.time() - t0:.0f}s): best 5 =",
      [(round(L, 4), c) for L, c in cands[:5]], flush=True)
obj_combo = {k: None for k in range(3)}
inits = {"best_halfbar": Nn[np.array(cands[0][1])[half], np.arange(M)], "mean": p @ Nn}
if len(set(cands[0][1])) > 1:                    # best structured map is not a single object: also start from argmax p
    inits[f"obj{np.argmax(p)+1}"] = Nn[np.argmax(p)]
# (the middle map is an affine image of the mean map up to 1e-3, so it is the same start for these affine-invariant losses)
res = {"loss": loss_name, "p": p.tolist(), "candidates": {}, "runs": {}, "halfbar_top5": [(L, list(c)) for L, c in cands[:5]]}
for k in range(3):
    res["candidates"][f"obj{k+1}"] = evaluate(Nn[k])
for nm, f0 in list(inits.items()) + [("middle", mid)]:
    res["candidates"][nm] = evaluate(f0)
t0 = time.time()
for nm, f0 in inits.items():
    f = torch.tensor(f0.copy(), requires_grad=True)
    opt = torch.optim.Adam([f], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        opt.zero_grad(); L = loss(f); L.backward(); opt.step(); sched.step()
    fn = f.detach().numpy()
    w, r = coeffs(fn)
    sp = split_of(to_img(fn, fg), st)
    gn = mad_norm(fn)
    share = [float(np.mean(np.abs(gn - Nn[k]) < 0.05)) for k in range(3)]
    res["runs"][nm] = dict(loss=evaluate(fn), split=np.round(sp, 4).tolist(), w=np.round(w / np.abs(w).sum(), 4).tolist(),
                           span_resid=round(float(r), 4), share_near_obj=[round(s, 3) for s in share])
    np.save(os.path.join(HERE, "opt", f"{loss_name}_{tag}_{nm}.npy"), fn)
    print(nm, res["runs"][nm], f"({time.time() - t0:.0f}s)", flush=True)
    res["runs"][nm]["profile"] = paper_profile(gn, st).tolist()     # paper's read-out of N(f), for check14
best = min(res["runs"], key=lambda k: res["runs"][k]["loss"])
res["best"] = best
print(json.dumps(res, indent=None))
with open(os.path.join(HERE, "opt", f"{loss_name}_{tag}.json"), "w") as fh:
    json.dump(res, fh, indent=1)
