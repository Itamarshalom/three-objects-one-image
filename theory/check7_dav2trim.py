"""Check 7 (docs/theory.md Sec. 3.3): Depth Anything V2's released students are trained on pseudo-labelled real images where
"for each pseudo-labeled sample, we ignore its top-n-largest-loss regions during training, where n is set as 10%"
(arXiv:2406.09414, Sec. 5.2).  Does 10% trimming of the ssi term change the object-only (regime A) optimum?

loss 'dav2trim' (per target k, averaged with weights p):
    mean of the 90% smallest |N(f) - N(d_k)|  +  2 * GM(N(f) - N(d_k))
(trimming applied per pixel to the ssi term only; whether DAv2 trims pixels or patches, and whether L_gm is used
on pseudo-labelled data, is not stated in the DAv2 paper; the variant 'dav2trimgm' also trims the GM residual image.)

Candidates: 3 objects, mean, middle map, all 729 half-bar maps, and the maps where the trimmed MiDaS loss beat
every object (check3b): the two-object patchwork found at p=(0.45,0.45,0.1) and the mean-like map at uniform p.
Then Adam (400 steps, as check3b) from: argmax object, the mean, the MiDaS two-object patchwork, the MiDaS
mean-like map.  The two MiDaS maps come from check3b_opt.py (opt/); if they are missing it is run first.
usage: python check7_dav2trim.py <p1,p2,p3> [iters] [loss]"""
import sys, os, time, json, itertools
import numpy as np, torch
from common import load, split_of, to_img, mad_norm, ensure, paper_profile

torch.set_num_threads(int(os.environ.get("NT", "4")))
torch.set_default_dtype(torch.float64)
p = np.array([float(x) for x in sys.argv[1].split(",")]); p = p / p.sum()
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 400
LOSS = sys.argv[3] if len(sys.argv) > 3 else "dav2trim"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "opt7"), exist_ok=True)
os.makedirs(os.path.join(HERE, "opt"), exist_ok=True)
for loss_p, start in [("0.45,0.45,0.1", "0.45_0.45_0.10_best_halfbar"), ("0.3333333,0.3333333,0.3333334", "0.33_0.33_0.33_mean")]:
    ensure([os.path.join("opt", f"midas_{start}.npy")], ["check3b_opt.py", "midas", loss_p],
           log=os.path.join("opt", f"log_midas_{loss_p}.txt"))

st = load("texture_a90")
fg = st["fg"]; H, W = fg.shape
X = st["depth"][:, fg]; M = X.shape[1]
Nn = np.array([mad_norm(X[k]) for k in range(3)])
idx = torch.tensor(np.flatnonzero(fg.ravel()))
mask = torch.tensor(fg, dtype=torch.float64)
Nt = torch.tensor(Nn)
KEEP = int(0.9 * M)


def N(f):
    m = f.median()
    return (f - m) / (f - m).abs().mean()


def img(v):
    out = torch.zeros(H * W)
    return out.index_put((idx,), v).view(H, W)


def gm(R):
    tot = 0
    for s in range(4):
        k = 2 ** s
        r, mk = R[::k, ::k], mask[::k, ::k]
        gx = (r[:, 1:] - r[:, :-1]).abs() * mk[:, 1:] * mk[:, :-1]
        gy = (r[1:, :] - r[:-1, :]).abs() * mk[1:, :] * mk[:-1, :]
        tot = tot + (gx.sum() + gy.sum()) / mk.sum()
    return tot


def loss(f):
    g = N(f); tot = 0
    for k in range(3):
        r = g - Nt[k]
        a = r.abs()
        keep = torch.topk(a, KEEP, largest=False)
        data = keep.values.mean()
        if LOSS == "dav2trimgm":                      # also zero the trimmed pixels in the GM residual
            m = torch.zeros(M); m[keep.indices] = 1.0
            r = r * m
        tot = tot + p[k] * (data + 2 * gm(img(r)))
    return tot


def evaluate(f):
    with torch.no_grad():
        return float(loss(torch.tensor(f)))


def coeffs(f):
    A = np.concatenate([X.T, np.ones((M, 1))], 1)
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    r = A @ c - f
    return c[:3], np.sqrt((r ** 2).mean()) / f.std()


tag = "_".join(f"{x:.2f}" for x in p)
mid = np.sort(Nn, 0)[1]
bar = np.minimum(st["cube"][fg] // (st["n"] // 3), 2)
half = bar * 2 + (mid < 0)
res = {"loss": LOSS, "p": p.tolist(), "candidates": {}, "runs": {}}
t0 = time.time()
for k in range(3):
    res["candidates"][f"obj{k+1}"] = evaluate(Nn[k])
res["candidates"]["mean"] = evaluate(p @ Nn)
res["candidates"]["middle"] = evaluate(mid)
patch2 = np.load(os.path.join(HERE, "opt", "midas_0.45_0.45_0.10_best_halfbar.npy"))   # MiDaS two-object patchwork
meanlike = np.load(os.path.join(HERE, "opt", "midas_0.33_0.33_0.33_mean.npy"))          # MiDaS mean-like optimum
res["candidates"]["midas_patchwork_0.45"] = evaluate(patch2)
res["candidates"]["midas_meanlike_unif"] = evaluate(meanlike)
cands = []
for combo in itertools.product(range(3), repeat=6):
    cands.append((evaluate(Nn[np.array(combo)[half], np.arange(M)]), combo))
cands.sort()
nonobj = [c for c in cands if len(set(c[1])) > 1]
res["halfbar_best"] = [cands[0][0], list(cands[0][1])]
res["halfbar_best_nonobject"] = [nonobj[0][0], list(nonobj[0][1])]
print(f"p={np.round(p,3)} {LOSS} candidates ({time.time()-t0:.0f}s):",
      {k: round(v, 4) for k, v in res["candidates"].items()},
      "| half-bar best", round(cands[0][0], 4), cands[0][1], "best non-object", round(nonobj[0][0], 4), nonobj[0][1],
      flush=True)

inits = {f"obj{np.argmax(p)+1}": Nn[np.argmax(p)], "mean": p @ Nn,
         "midas_patchwork_0.45": patch2, "midas_meanlike_unif": meanlike}
for nm, f0 in inits.items():
    f = torch.tensor(f0.copy(), requires_grad=True)
    opt = torch.optim.Adam([f], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        opt.zero_grad(); L = loss(f); L.backward(); opt.step(); sched.step()
    fn = f.detach().numpy()
    w, r = coeffs(fn)
    sp = split_of(to_img(fn, fg), st)
    res["runs"][nm] = dict(start=evaluate(f0), loss=evaluate(fn), split=np.round(sp, 4).tolist(),
                           w=np.round(w / np.abs(w).sum(), 4).tolist(), span_resid=round(float(r), 4))
    np.save(os.path.join(HERE, "opt7", f"{LOSS}_{tag}_{nm}.npy"), fn)
    print(f"  Adam from {nm}: {res['runs'][nm]} ({time.time()-t0:.0f}s)", flush=True)
    res["runs"][nm]["profile"] = paper_profile(mad_norm(fn), st).tolist()     # paper's read-out of N(f), for check14
best_obj = min(res["candidates"][f"obj{k+1}"] for k in range(3))
best_run = min(v["loss"] for v in res["runs"].values())
res["verdict"] = "object wins" if best_obj <= best_run + 1e-4 else "non-object map wins"
print(f"  best object {best_obj:.4f} vs best Adam end {best_run:.4f}: {res['verdict']}", flush=True)
json.dump(res, open(os.path.join(HERE, "opt7", f"{LOSS}_{tag}.json"), "w"), indent=1)
