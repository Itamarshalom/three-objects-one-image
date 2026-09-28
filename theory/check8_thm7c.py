"""Check 8 (Theorem 7(c), docs/theory.md Sec. 3.2): SSI-MAE (median/MAD) optimum below p_k = 1/2 in the symmetric continuous
model n_k(theta) = T(1/2 - frac(theta - theta_k)), theta_k = (k-1)/3, T = 4 (so every object has MAD 1), h = T/3.

Problem: min_g C(g) = sum_k p_k mean|g - n_k|  s.t. mean|g| = 1, median g = 0.
Same-side construction (printed as "old construction"): only 'same-side' moves (middle value -> outer value on the same side of 0) are used.
Lagrangian used here: every half-bar is an ordered pair (c, c') of corners (c adjacent, c' at the
far end of the bar); at distance s in [0,1) from the bar midpoint (s = 1 at corner c) a pixel either
  stays at the middle value                                  (0),
  takes object c   (same side,  MAD gain h,       cost a_c h),  or
  takes object c'  (crosses 0,  MAD gain (1-s) h, cost a_c' h),     a_k := 1 - 2 p_k,
choosing max(0, lam - a_c, lam (1 - s) - a_c').  Budget: total gain over the 6 half-bars = 3 (in units h/6).
We check: (1) the closed-form dual equals the brute-force dual over candidates {n_1, n_2, n_3, 0};
(2) a dual that also prices the median constraint; (3) a feasible primal (Lagrangian map + partial fill, then
median/MAD normalised) against both bounds; (4) the old construction's cost; (5) object regions."""
import numpy as np

T = 4.0; h = T / 3; K = 120000
th = (np.arange(K) + 0.5) / K
Nn = np.array([T * (0.5 - np.mod(th - k / 3, 1)) for k in range(3)])
mid = np.sort(Nn, 0)[1]
cand = np.concatenate([Nn, np.zeros((1, K))]); absv = np.abs(cand)


def C(g, p):
    return sum(p[k] * np.abs(g - Nn[k]).mean() for k in range(3))


def Nz(g):
    m = np.median(g); return (g - m) / np.abs(g - m).mean()


def old_claim(p):
    """the same-side construction: middle map, then whole same-side half-bars of objects in order of p"""
    g = mid.copy()
    for j in np.argsort(-p):
        need = 1 - np.abs(g).mean()
        if need <= 1e-12: break
        same = (np.sign(Nn[j]) == np.sign(mid)) & (np.abs(Nn[j]) > np.abs(mid) + 1e-9) & (g == mid)
        ids = np.flatnonzero(same); gain = (np.abs(Nn[j]) - np.abs(mid))[ids]
        cs = np.cumsum(gain) / K
        take = ids[: np.searchsorted(cs, need) + 1] if cs[-1] > need else ids
        g[take] = Nn[j][take]
    return g


def psi(lam, ac, acp, ns=20001):
    s = (np.arange(ns) + 0.5) / ns
    return np.maximum(0, np.maximum(lam - ac, lam * (1 - s) - acp)).mean()


def dual_closed(lam, p):
    a = 1 - 2 * np.asarray(p)
    Cmid = 2 * h / 3
    return Cmid + lam * h / 2 - h / 6 * sum(psi(lam, a[c], a[cp]) for c in range(3) for cp in range(3) if c != cp)


def dual_brute(lam, p, mup=0.0, mun=0.0):
    cost = np.array([sum(p[k] * np.abs(cand[j] - Nn[k]) for k in range(3)) for j in range(4)])
    pen = cost - lam * absv + mup * (cand > 1e-12) + mun * (cand < -1e-12)
    return lam - (mup + mun) / 2 + pen.min(0).mean(), pen


def primal(p, lam, mup=0.0, mun=0.0, eps=1e-4):
    """Lagrangian minimiser at lam, indifferent pixels filled (cheapest first) until MAD = 1, then N()"""
    _, pen_lo = dual_brute(lam - eps, p, mup, mun)
    _, pen_hi = dual_brute(lam + eps, p, mup, mun)
    lo, hi = pen_lo.argmin(0), pen_hi.argmin(0)
    g = cand[lo, np.arange(K)].copy()
    need = K * (1 - np.abs(g).mean())
    for i in np.flatnonzero(lo != hi):
        if need <= 0: break
        need -= absv[hi[i], i] - absv[lo[i], i]; g[i] = cand[hi[i], i]
    return g


PS = [(0.48, 0.30, 0.22), (0.45, 0.45, 0.10), (0.40, 0.35, 0.25), (0.36, 0.33, 0.31), (0.49, 0.26, 0.25),
      (0.42, 0.42, 0.16), (1/3, 1/3, 1/3)]
lams = np.linspace(0.0, 0.999, 1999)
print("p | old construction | closed-form dual max (lam*) | brute dual (lam only) | dual with median prices | "
      "feasible primal N(g_lam*) | gap | share of loop overwritten by object k (g = n_k != middle)")
for p in PS:
    p = np.array(p)
    Dc = np.array([dual_closed(l, p) for l in lams]); ls = lams[Dc.argmax()]
    Db = max(dual_brute(l, p)[0] for l in lams[::4])
    # median-priced dual on a coarse (lam, mu+, mu-) grid around lam*
    best = (-1e9, None)
    for l in np.linspace(max(0, ls - 0.05), min(0.999, ls + 0.05), 21):
        for mp in np.linspace(0, 0.3, 7):
            for mn in np.linspace(0, 0.3, 7):
                v = dual_brute(l, p, mp, mn)[0]
                if v > best[0]: best = (v, (l, mp, mn))
    g = primal(p, ls)
    gN = Nz(g)
    Cp = C(gN, p)
    cov = [np.mean(np.isclose(g, Nn[k], atol=1e-9) & ~np.isclose(Nn[k], mid)) for k in range(3)]
    print(f"{np.round(p,3)} | {C(old_claim(p), p):.5f} | {Dc.max():.5f} ({ls:.3f}) | {Db:.5f} | {best[0]:.5f} | "
          f"{Cp:.5f} (med {np.median(g):+.4f}, MAD {np.abs(g).mean():.4f}) | {Cp - max(Dc.max(), best[0]):.5f} | "
          f"{np.round(cov, 3)}  (C(n_argmax) = {C(Nn[np.argmax(p)], p):.5f})")

print("\nregion structure at lam*: object k occupies its two adjacent half-bars (if lam* >= a_k) plus, on each far "
      "half-bar, the band s < 1 - a_k/lam* next to the bar midpoint (unless a more probable object outbids it)")
for p in [(0.48, 0.30, 0.22), (0.40, 0.35, 0.25), (0.49, 0.26, 0.25)]:
    p = np.array(p); a = 1 - 2 * p
    Dc = np.array([dual_closed(l, p) for l in lams]); ls = lams[Dc.argmax()]
    print(f"p={np.round(p,3)} lam*={ls:.3f}: band fraction of the far half-bars 1 - a_k/lam* =",
          np.round(np.clip(1 - a / ls, 0, 1), 3), " (-> 1 as p_k -> 1/2: object k then covers both adjacent bars,"
          " i.e. g = n_k on the whole loop because n_k = middle on the third bar)")
# continuity: p_1 -> 1/2
for p1 in [0.40, 0.45, 0.48, 0.49, 0.499]:
    p = np.array([p1, (1 - p1) * 0.55, (1 - p1) * 0.45])
    Dc = np.array([dual_closed(l, p) for l in lams]); ls = lams[Dc.argmax()]
    gN = Nz(primal(p, ls))
    g = primal(p, ls)
    print(f"p1={p1}: lam*={ls:.3f} optimum cost {C(gN, p):.5f} vs object 1 {C(Nn[0], p):.5f}; share of loop where g = n_1 "
          f"{np.mean(np.isclose(g, Nn[0], atol=1e-9)):.3f} (object 1 alone: 1.000)")
