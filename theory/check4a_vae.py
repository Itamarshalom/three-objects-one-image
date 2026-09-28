"""Check 4a (docs/theory.md Sec. 4a-4b): geometry of the three objects in Marigold v1-0's latent space and how linear the VAE decoder is
along their convex hull.  Uses the locally cached prs-eth/marigold-depth-v1-0 VAE (SD2 VAE), CPU, fp32.

Depth targets are built the way Marigold normalises training depth: a full image (object + a far back wall),
2/98 percentiles over all pixels -> [-1, 1], replicated to 3 channels, encoded (latent mean * scaling factor).
Two backgrounds: 'wall' = back wall at (max fg depth + T); 'tight' = back wall right behind the object
(max fg depth + 0.5), so the object fills the depth range (the 'tighter crop' plan)."""
import os, json
import numpy as np, torch
from diffusers import AutoencoderKL
from common import load, split_of

torch.set_num_threads(int(os.environ.get("NT", "4")))
st = load("texture_a90")
fg = st["fg"]; D = st["depth"]
T = 5 * np.sqrt(3)
vae = AutoencoderKL.from_pretrained("prs-eth/marigold-depth-v1-0", subfolder="vae", variant="fp16",
                                    torch_dtype=torch.float16).to(torch.float32).eval()
sf = vae.config.scaling_factor
print("VAE scaling factor", sf)


def target(k, bg_gap):
    d = np.where(fg, D[k], np.nanmax(D[:, fg]) + bg_gap)       # same wall for all three objects
    a, b = np.percentile(d, [2, 98])
    return np.clip(((d - a) / (b - a) - 0.5) * 2, -1.5, 1.5)


@torch.no_grad()
def enc(d):
    x = torch.tensor(d, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
    return vae.encode(x).latent_dist.mean * sf


@torch.no_grad()
def dec(z):
    return vae.decode(z / sf).sample.mean(1)[0].numpy()


out = {}
for bg_name, gap in [("wall", T), ("tight", 0.5)]:
    tg = np.array([target(k, gap) for k in range(3)])
    Z = torch.cat([enc(tg[k]) for k in range(3)])                  # [3, 4, 64, 64]
    Zf = Z.reshape(3, -1).double().numpy()
    Gz = Zf @ Zf.T
    dist = np.sqrt(np.array([[((Zf[i] - Zf[j]) ** 2).sum() for j in range(3)] for i in range(3)]))
    fg_range = np.ptp(tg[0][fg])
    print(f"\n[{bg_name}] object occupies {fg_range / 2:.2f} of the [-1,1] range; latent dim {Zf.shape[1]}")
    print("  ||z_k|| =", np.round(np.sqrt(np.diag(Gz)), 2), " pairwise ||z_j - z_k|| =", np.round(dist[np.triu_indices(3, 1)], 2))
    # decoder checks: reconstruction of each object and decoded convex combinations
    rec = [dec(Z[k:k + 1]) for k in range(3)]
    for k in range(3):
        print(f"  Dec(z_{k+1}) split = {np.round(split_of(np.where(fg, rec[k], np.nan), st), 3)}")
    rows = []
    for w in [(0.5, 0.5, 0), (1/3, 1/3, 1/3), (0.6, 0.3, 0.1), (0.4, 0.35, 0.25), (0.8, 0.15, 0.05)]:
        w = np.array(w)
        zm = sum(w[k] * Z[k:k + 1] for k in range(3))
        dm = dec(zm)
        s_dec_of_mean = split_of(np.where(fg, dm, np.nan), st)
        s_mean_of_dec = split_of(np.where(fg, sum(w[k] * rec[k] for k in range(3)), np.nan), st)
        err = np.abs(dm - sum(w[k] * rec[k] for k in range(3)))[fg]
        print(f"  w={np.round(w, 3)}: split Dec(sum w z) = {np.round(s_dec_of_mean, 3)} | split sum w Dec(z) = "
              f"{np.round(s_mean_of_dec, 3)} | |Dec(mean) - mean(Dec)| on fg: mean {err.mean():.4f}, max {err.max():.3f} "
              f"(object's own depth range {fg_range:.3f})")
        rows.append(dict(w=w.tolist(), split_dec_mean=s_dec_of_mean.tolist(), split_mean_dec=s_mean_of_dec.tolist()))
    out[bg_name] = dict(gram=Gz.tolist(), dist=dist.tolist(), fg_range=float(fg_range), rows=rows)
    np.save(f"latents_{bg_name}.npy", Zf)
json.dump(out, open("vae_geometry.json", "w"), indent=1)
