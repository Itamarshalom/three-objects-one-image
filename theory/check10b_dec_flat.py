"""Check 10b (docs/theory.md Sec. 4b): decoder bias phi(w) of the Marigold v1-0 VAE at the flatness of the pilot outputs (object 12% of the
frame range, wall gap 7.8 T). Split of Dec(sum_k w_k z_k) against w."""
import os, sys
import numpy as np, torch
THEORY = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THEORY)
from diffusers import AutoencoderKL
from common import load, split_of
torch.set_num_threads(4)
st = load("texture_a90"); fg = st["fg"]; D = st["depth"]; T = 5 * np.sqrt(3)
vae = AutoencoderKL.from_pretrained("prs-eth/marigold-depth-v1-0", subfolder="vae", variant="fp16",
                                    torch_dtype=torch.float16).to(torch.float32).eval()
sf = vae.config.scaling_factor
def target(k, gap):
    d = np.where(fg, D[k], np.nanmax(D[:, fg]) + gap)
    a, b = np.percentile(d, [2, 98])
    return np.clip(((d - a) / (b - a) - 0.5) * 2, -1.5, 1.5)
@torch.no_grad()
def enc(d):
    return vae.encode(torch.tensor(d, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)).latent_dist.mean * sf
@torch.no_grad()
def dec(z):
    return vae.decode(z / sf).sample.mean(1)[0].numpy()
for gapT in [7.8]:
    tg = np.array([target(k, gapT * T) for k in range(3)])
    Z = torch.cat([enc(tg[k]) for k in range(3)])
    for k in range(3):
        print(f"gap {gapT}T Dec(z_{k+1}) split {np.round(split_of(np.where(fg, dec(Z[k:k+1]), np.nan), st), 3)}", flush=True)
    for w in [(0.5, 0.5, 0), (1/3, 1/3, 1/3), (0.6, 0.3, 0.1), (0.4, 0.35, 0.25)]:
        w = np.array(w); zm = sum(w[k] * Z[k:k + 1] for k in range(3))
        s = split_of(np.where(fg, dec(zm), np.nan), st)
        print(f"gap {gapT}T w={np.round(w,3)} split Dec(sum w z)={np.round(s,3)} max|dev|={np.abs(s-w).max():.3f}", flush=True)
