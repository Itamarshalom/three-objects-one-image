"""Check 11: the exact-denoiser analysis of Sec. 4a/4c under Marigold's TRAINING noise law.

Marigold v1-0 is trained with annealed multi-resolution noise (prs-eth/Marigold: src/util/multi_res_noise.py,
config multi_res_noise {strength 0.9, annealed true, downscale_strategy original}; trainer: strength * t / 1000),
but diffusers samples x_T ~ N(0, I).  Gaussian approximation of the training noise at step t: n ~ N(0, Sigma_t),
Sigma_t = E over the random factors r of the per-channel spatial covariance of multi_res_noise_like, divided by its
mean diagonal (the code divides by noise.std()).  The Bayes denoiser trained under that law has
    log pi_k = log p_k + (a/s2) x^T Sigma_t^-1 z_k - (a^2 / 2 s2) z_k^T Sigma_t^-1 z_k .
On iid input x_T ~ N(0,I) its log-odds noise has sd (a/s2) ||Sigma_T^-1 (z_j - z_k)|| ('kappa_MR').
DDIM (eta = 0, trailing, 50 steps) keeps x_t = alpha_t x_T + Z beta_t, so the whole run needs only the Gaussian
features x_T^T Sigma_t^-1 z_k (jointly over all steps) and the 3x3 matrices Z Sigma_t^-1 Z^T: exact under the
Gaussian approximation.  Sanity: with Sigma_t = I it must reproduce check4b.
Output: kappa (iid formula, MR), seed-mean one-step weights s, their sd, q at 50 steps, P(final = 1-step argmax),
validity at 1 step, for the latents of check4a/check10 (object share 6.7%-106% of the depth range)."""
import os, json
from collections import Counter
import numpy as np, torch
from common import ensure

HERE = os.path.dirname(os.path.abspath(__file__))
GAPS = ["gap0.5abs", "gap1T", "gap3.2T", "gap7.8T", "gap15T"]
ensure([f"latents_{nm}.npy" for nm in GAPS], ["check10_kappa_flat.py"])   # latents written by check10
H = W = 64; n = H * W
betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
abar = np.cumprod(1 - betas)
rng = np.random.default_rng(0)

# ---- level statistics of the 'original' downscale strategy (same arithmetic as the Marigold code)
tally = Counter(); ND = 50000
for _ in range(ND):
    w = H
    for i in range(10):
        r = rng.random() * 2 + 2
        w = max(1, int(w / r ** i))
        tally[(i, w)] += 1
        if w == 1:
            break


def up_matrix(w):
    eye = torch.eye(w * w, dtype=torch.float64).reshape(w * w, 1, w, w)
    U = torch.nn.functional.interpolate(eye, size=(H, W), mode="bilinear")   # == nn.Upsample(bilinear) default
    return U.reshape(w * w, n).T.numpy()


def up1(w):                                                                   # 1-d bilinear upsampling w -> 64
    eye = torch.eye(w, dtype=torch.float64).reshape(w, 1, w)
    return torch.nn.functional.interpolate(eye, size=H, mode="linear").reshape(w, H).T.numpy()


for wt in [5, 17]:                                                          # bilinear 2-d = Kronecker of 1-d
    assert np.allclose(up_matrix(wt), np.kron(up1(wt), up1(wt)), atol=1e-12)
L = {1: np.zeros((n, n)), 2: np.zeros((n, n)), 3: np.zeros((n, n))}
for (i, w), c in tally.items():
    if i == 0:
        continue                                                            # i = 0: a second full-res N(0, I)
    u = up1(w); uu = u @ u.T
    L[i] += (c / ND) * np.kron(uu, uu)
print("levels (i, w): share of draws", sorted((k, round(v / ND, 3)) for k, v in tally.items() if k[0] > 0), flush=True)


def Sigma(s):
    S = 2.0 * np.eye(n) + s ** 2 * L[1] + s ** 4 * L[2] + s ** 6 * L[3]
    return S / np.mean(np.diag(S))


# ---- Monte Carlo check of Sigma_T against the actual Marigold function (verbatim logic, batch 2 x 4 channels)
def multi_res_noise_like(shape, strength, gen):
    b, c, w, h = shape
    up = torch.nn.Upsample(size=(w, h), mode="bilinear")
    noise = torch.randn(shape, generator=gen, dtype=torch.float64)
    for i in range(10):
        r = torch.rand(1, generator=gen, dtype=torch.float64) * 2 + 2
        w, h = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        noise += up(torch.randn(b, c, w, h, generator=gen, dtype=torch.float64)) * strength ** i
        if w == 1 or h == 1:
            break
    return noise / noise.std()


sT = 0.9 * 999 / 1000
ST = Sigma(sT)
ev = np.linalg.eigvalsh(ST)
print(f"Sigma_T (per channel): eigenvalues {ev.min():.3f} .. {ev.max():.1f}", flush=True)
Zs = {nm: np.load(os.path.join(HERE, f"latents_{nm}.npy")).reshape(3, 4, n) for nm in GAPS}
dirs = {"z1-z2 (gap1T), ch0": Zs["gap1T"][0, 0] - Zs["gap1T"][1, 0], "constant": np.ones(n),
        "random iid": rng.standard_normal(n)}
gen = torch.Generator().manual_seed(0)
draws = torch.cat([multi_res_noise_like((2, 4, H, W), sT, gen).reshape(8, n) for _ in range(1000)]).numpy()
for nm, v in dirs.items():
    v = v / np.linalg.norm(v)
    print(f"  MC check along {nm:22s}: v'Sigma v model {v @ ST @ v:8.3f} | Monte Carlo {np.var(draws @ v):8.3f}", flush=True)
del draws

# ---- DDIM with the (Gaussian-approximate) MR-trained Bayes denoiser, iid start
from scipy.linalg import cho_factor, cho_solve
S = 50
ts = np.round(np.arange(1000, 0, -1000 / S)).astype(int) - 1
SiZ = {nm: [] for nm in Zs}                               # per step: Sigma_t^-1 z_k per channel, [4, n, 3]
SiD_T = {}
for t in ts:
    cf = cho_factor(Sigma(0.9 * t / 1000))
    for nm, Z in Zs.items():
        Zc = Z.transpose(1, 2, 0)                          # [4, n, 3]
        SiZ[nm].append(np.stack([cho_solve(cf, Zc[c]) for c in range(4)]))
print("Sigma_t^-1 Z computed for the 50 trailing steps", flush=True)


def run(Z, p, law, cur, nseed=20000, seed=1):
    """Z: [3, 4, n] latents.  law 'iid' (Sigma = I) or 'mr'."""
    Zc = Z.transpose(1, 2, 0)                             # [4, n, 3]
    B, Mt = [], []
    for i_t, t in enumerate(ts):
        Bt = Zc.copy() if law == "iid" else SiZ[cur][i_t]
        B.append(Bt.reshape(4 * n, 3))
        Mt.append(np.einsum("cnk,cnl->kl", Zc, Bt))
    Ball = np.concatenate(B, 1)                           # [4n, 3S]
    Cov = Ball.T @ Ball
    ev_, V_ = np.linalg.eigh(Cov)
    Lf = V_ * np.sqrt(np.clip(ev_, 0, None))
    g = np.random.default_rng(seed).standard_normal((nseed, 3 * S)) @ Lf.T    # features x_T^T Sigma_t^-1 z_k
    g = g.reshape(nseed, S, 3)
    logp = np.log(p); alpha = np.ones(nseed); beta = np.zeros((nseed, 3)); first = None
    for i, t in enumerate(ts):
        a, s2 = np.sqrt(abar[t]), 1 - abar[t]
        lin = alpha[:, None] * g[:, i] + beta @ Mt[i]
        lg = logp + (a * lin - a ** 2 * np.diag(Mt[i]) / 2) / s2
        lg -= lg.max(1, keepdims=True); pi = np.exp(lg); pi /= pi.sum(1, keepdims=True)
        if first is None:
            first = pi.copy()
        ap = abar[ts[i + 1]] if i + 1 < len(ts) else abar[0]
        c1 = np.sqrt(1 - ap) / np.sqrt(s2)
        beta = np.sqrt(ap) * pi + c1 * (beta - a * pi)
        alpha = c1 * alpha
    ch = pi.argmax(1)
    return dict(q=np.bincount(ch, minlength=3) / nseed, s=first.mean(0), sd=first.std(0),
                agree=float((first.argmax(1) == ch).mean()), valid1=float((first.max(1) > 0.99).mean()),
                valid=float((pi.max(1) > 0.99).mean()), hot=float((first.max(1) > 0.9).mean()))


a, s2 = np.sqrt(abar[-1]), 1 - abar[-1]
flat = json.load(open(os.path.join(HERE, "vae_geometry_flat.json")))
res = {}
for nm, Z in Zs.items():
    kid, kmr = [], []
    for j, k in [(0, 1), (0, 2), (1, 2)]:
        D = Z[j] - Z[k]
        kid.append(a / s2 * np.linalg.norm(D))
        kmr.append(a / s2 * np.linalg.norm(SiZ[nm][0][:, :, j] - SiZ[nm][0][:, :, k]))
    print(f"\n[{nm}] object share of [-1,1] {flat[nm]['share']:.3f} | kappa iid-formula {np.round(kid, 2)} | "
          f"kappa MR-trained, iid input {np.round(kmr, 2)}", flush=True)
    for p in [(0.6, 0.3, 0.1), (0.4, 0.35, 0.25)]:
        for law in ["iid", "mr"]:
            r = run(Z, np.array(p), law, nm)
            res[f"{nm}|{p}|{law}"] = {k: np.asarray(v).tolist() for k, v in r.items()}
            print(f"  p={p} {law:3s}: seed-mean 1-step s {np.round(r['s'], 3)} sd {np.round(r['sd'], 3)} | q(50) "
                  f"{np.round(r['q'], 3)} | max|q-s| {np.abs(r['q'] - r['s']).max():.3f} | P(final=1-step argmax) "
                  f"{r['agree']:.3f} | valid at 1 step {r['valid1']:.3f} (max weight > 0.9: {r['hot']:.2f}) | valid at 50 {r['valid']:.2f}",
                  flush=True)
json.dump(res, open(os.path.join(HERE, "mrnoise_ddim.json"), "w"), indent=0)
