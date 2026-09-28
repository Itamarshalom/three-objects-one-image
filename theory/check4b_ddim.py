"""Check 4b (docs/theory.md Sec. 4c): DDIM (eta=0) with the EXACT denoiser of a 3-atom posterior, to see what the q-vs-p test can show.

Posterior over clean latents: z_k with prob p_k.  Exact denoiser: E[z | x_t] = sum_k pi_k(x_t) z_k,
pi_k ∝ p_k exp((a_t <x_t, z_k> - a_t^2 |z_k|^2 / 2) / s_t^2), a_t = sqrt(abar_t), s_t^2 = 1 - abar_t.
Everything depends on x_t only through <x_t, z_k>, so we simulate exactly in the 3-d span of the atoms
(coordinates from the Gram matrix).  SD2 scaled-linear schedule (Marigold v1-0), trailing or leading spacing.
Two starts: x_T ~ N(0, I) (what samplers do) and x_T ~ p_T (the true forward marginal, where q = p must hold).
Geometries: the Marigold latents of vae_geometry.json (written by check4a; left out if the file is missing) and
idealised equilateral atoms."""
import json
import numpy as np

betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
abar = np.cumprod(1 - betas)
print(f"SD scaled-linear: abar_999 = {abar[-1]:.5f}, sqrt = {np.sqrt(abar[-1]):.5f}, SNR_T = {abar[-1] / (1 - abar[-1]):.5f}, "
      f"abar_0 = {abar[0]:.5f}")


def timesteps(S, spacing):
    if spacing == "trailing":
        ts = np.round(np.arange(1000, 0, -1000 / S)).astype(int) - 1
    else:                                               # leading, steps_offset = 1 (the pre-fix Marigold default)
        ts = (np.arange(0, S) * (1000 // S)).round()[::-1].astype(int) + 1
    return ts


def denoise(u, t, C, logp):
    a, s2 = np.sqrt(abar[t]), 1 - abar[t]
    logits = logp + (a * u @ C - a ** 2 * (C ** 2).sum(0) / 2) / s2      # C: [3 coords, 3 atoms]
    logits -= logits.max(1, keepdims=True)
    pi = np.exp(logits); pi /= pi.sum(1, keepdims=True)
    return pi


def run(G, p, S, spacing="trailing", start="noise", n=20000, seed=0):
    rng = np.random.default_rng(seed)
    C = np.linalg.cholesky(G + 1e-9 * np.eye(3)).T                      # columns = atom coordinates, C^T C = G
    logp = np.log(p)
    ts = timesteps(S, spacing)
    if start == "noise":
        u = rng.standard_normal((n, 3))
    else:
        k = rng.choice(3, n, p=p)
        u = np.sqrt(abar[ts[0]]) * C[:, k].T + np.sqrt(1 - abar[ts[0]]) * rng.standard_normal((n, 3))
    pi_first = None
    for i, t in enumerate(ts):
        pi = denoise(u, t, C, logp)
        if pi_first is None:
            pi_first = pi.copy()
        x0 = pi @ C.T
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else abar[0]
        eps = (u - np.sqrt(abar[t]) * x0) / np.sqrt(1 - abar[t])
        u = np.sqrt(a_prev) * x0 + np.sqrt(1 - a_prev) * eps
    choice = pi.argmax(1)
    q = np.bincount(choice, minlength=3) / n
    return dict(q=q, valid=float((pi.max(1) > 0.99).mean()), mean_pi_first=pi_first.mean(0),
                agree_first=float((pi_first.argmax(1) == choice).mean()), pi_first_sd=pi_first.std(0))


def kappa(G):
    d2 = np.array([G[i, i] + G[j, j] - 2 * G[i, j] for i, j in [(0, 1), (0, 2), (1, 2)]])
    return np.sqrt(abar[-1]) * np.sqrt(d2) / (1 - abar[-1])


PS = [np.array(x) for x in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (1/3, 1/3, 1/3)]]
res = {}
geo = {}
try:
    vg = json.load(open("vae_geometry.json"))
    for k, v in vg.items():
        geo[f"Marigold latent, {k}"] = np.array(v["gram"])
except FileNotFoundError:
    print("vae_geometry.json missing: run check4a_vae.py first; using idealised geometries only")
# idealised: equilateral atoms (pairwise distance r) sharing a common component of norm c (in units of the noise std)
for r in [1, 5, 15, 40]:
    c2 = 100.0 ** 2
    G = np.full((3, 3), c2) + np.eye(3) * r ** 2 / 2
    geo[f"ideal |dz|={r}"] = G

for name, G in geo.items():
    print(f"\n== {name}: pairwise kappa = sqrt(abar_T)|z_j - z_k| / (1 - abar_T) = {np.round(kappa(G), 3)}, "
          f"|z_k| = {np.round(np.sqrt(np.diag(G)), 1)}")
    for p in PS:
        for start in ["noise", "pT"]:
            line = []
            for S in [1, 4, 10, 50]:
                r = run(G, p, S, start=start)
                line.append(f"S={S:2d}: q={np.round(r['q'], 3)} valid={r['valid']:.2f}")
                res[f"{name}|{np.round(p, 3).tolist()}|{start}|{S}"] = {k: np.asarray(v).tolist() for k, v in r.items()}
            r1 = run(G, p, 50, start=start)
            print(f"  p={np.round(p, 3)} start={start:5s} | mean 1-step weights={np.round(r1['mean_pi_first'], 3)} "
                  f"(sd {np.round(r1['pi_first_sd'], 3)}) | P(final = argmax 1-step)={r1['agree_first']:.3f} | " + " | ".join(line))

# leading spacing with the exact denoiser (1 step evaluates the denoiser at t=1 on pure noise)
G = geo[list(geo)[0]]
for S in [1, 4, 50]:
    r = run(G, PS[0], S, spacing="leading")
    print(f"leading spacing, {list(geo)[0]}, p={PS[0]}, S={S} (timesteps {timesteps(S, 'leading')[:3]}...): q={np.round(r['q'], 3)}, "
          f"valid={r['valid']:.2f}")
json.dump(res, open("ddim_exact.json", "w"), indent=0)
