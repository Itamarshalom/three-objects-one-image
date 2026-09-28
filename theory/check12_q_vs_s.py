"""Check 12 (docs/theory.md Sec. 4c): is 'q = seed-averaged one-step weights (within 0.03)' a property of exact samplers, or of
the geometries tried in check4b?  Exact 3-atom denoiser, trailing DDIM, x_T ~ N(0, I), SD2 schedule (check4b code).
Random Marigold-like atoms: common component so that |z_k| = 167 +/- 0.6 (measured spread: wall 0.2, tight 1.15),
the distance between atoms 1 and 2 drawn from U(5, 90), the other two pairs free (pairwise kappa 0.2-18.7 over the
80 sets, 0.4-6.3 for atoms 1 and 2), p ~ Dirichlet(2,2,2).  Same atoms with the three norms equalised.
Then the encoded geometries of vae_geometry_flat.json (written by check10).
Also: the total-variation bound of Prop. 9 -- the offset of p_T's mean from 0 along the common direction."""
import os, json
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "check4b_ddim.py"), encoding="utf-8").read().split("PS = [")[0]
ns = {}
exec(src, ns)
run, kappa, abar = ns["run"], ns["kappa"], ns["abar"]
rng = np.random.default_rng(5)
rows = []
for trial in range(80):
    d = rng.uniform(5, 90)
    off = rng.standard_normal((3, 3)); off -= off.mean(0); off *= d / np.sqrt(((off[0] - off[1]) ** 2).sum())
    Z = np.zeros((3, 4)); Z[:, 1:] = off
    tn = 167.0 + rng.uniform(-0.6, 0.6, 3)
    Z[:, 0] = np.sqrt(np.maximum(tn ** 2 - (off ** 2).sum(1), 1))
    p = rng.dirichlet([2, 2, 2])
    r = run(Z @ Z.T, p, 50)
    Z2 = Z.copy(); Z2[:, 0] = np.sqrt(167.0 ** 2 - (Z[:, 1:] ** 2).sum(1))
    r2 = run(Z2 @ Z2.T, p, 50)
    rows.append(dict(dqs=np.abs(r["q"] - r["mean_pi_first"]).max(), dqs_eq=np.abs(r2["q"] - r2["mean_pi_first"]).max(),
                     dqp=np.abs(r["q"] - p).max(), kmin=kappa(Z @ Z.T).min(), kmax=kappa(Z @ Z.T).max(),
                     p=p.round(2), q=r["q"].round(3), s=r["mean_pi_first"].round(3),
                     spread=np.ptp(np.linalg.norm(Z, axis=1)), move=np.abs(Z2 - Z).max()))
a = np.array([r["dqs"] for r in rows]); b = np.array([r["dqs_eq"] for r in rows])
km = np.array([r["kmin"] for r in rows])
print(f"80 random Marigold-like geometries: max|q-s| median {np.median(a):.3f}, 90th pct {np.percentile(a, 90):.3f}, "
      f"max {a.max():.3f}, share > 0.03: {np.mean(a > 0.03):.2f}")
print(f"  same atoms, norms equalised (largest coordinate move {max(r['move'] for r in rows):.2f}): max|q-s| median "
      f"{np.median(b):.3f}, max {b.max():.3f}, share > 0.03: {np.mean(b > 0.03):.2f}")
print(f"  share > 0.03 among min-kappa < 1: {np.mean(a[km < 1] > 0.03):.2f} (n={int((km < 1).sum())}); "
      f"among min-kappa >= 1: {np.mean(a[km >= 1] > 0.03):.2f} (n={int((km >= 1).sum())})")
for r in sorted(rows, key=lambda r: -r["dqs"])[:4]:
    print(f"  worst: max|q-s| {r['dqs']:.3f} (equal norms {r['dqs_eq']:.3f}) kappa {r['kmin']:.2f}-{r['kmax']:.2f} "
          f"norm spread {r['spread']:.2f} p {r['p']} q {r['q']} s {r['s']}")
# encoded ground-truth geometries (check4a / check10)
flat = json.load(open(os.path.join(HERE, "vae_geometry_flat.json")))
for nm, g in flat.items():
    G = np.array(g["gram"])
    out = []
    for p in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (0.5, 0.3, 0.2)]:
        r = run(G, np.array(p), 50)
        out.append(f"p={p}: q {np.round(r['q'], 3)} s {np.round(r['mean_pi_first'], 3)} max|q-s| "
                   f"{np.abs(r['q'] - r['mean_pi_first']).max():.3f}")
    print(f"encoded GT {nm:9s} (share {g['share']:.3f}, norm spread {np.ptp(np.sqrt(np.diag(G))):.2f}): " + " | ".join(out))
# Prop. 9's TV bound: p_T's mean sits a_T |zbar| / sigma_T noise-sd away from 0 along the common direction
G = np.array(flat["gap1T"]["gram"]); zbar = np.sqrt(G.mean())
off = np.sqrt(abar[-1]) * zbar / np.sqrt(1 - abar[-1])
print(f"a_T |zbar| / sigma_T = {off:.1f} -> TV(N(0,1), N({off:.1f},1)) = {2 * norm.cdf(off / 2) - 1:.6f} (the bound is vacuous)")
