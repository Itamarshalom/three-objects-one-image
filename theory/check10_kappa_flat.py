"""Check 10 (docs/theory.md Sec. 4a): kappa for Marigold v1-0 as a function of how much of the frame's depth range the object
occupies.  The pilot's own Marigold outputs give the object only 7-27% of the frame's 2/98 range (per-cube profile
ptp 3-15%), while check4a used 55% ('wall') and 106% ('tight').  Same construction as check4a (whole frame, one
back wall, 2/98 percentiles -> [-1,1], 3 channels, SD2 VAE latent mean x 0.18215), wall gaps from 0.5 (abs) to 15 T.
Saves latents_gap*.npy for check11 and prints the exact-denoiser predictions (iid noise law, check4b code):
seed-to-seed sd of the 1-step weight on object 1, P(final object = 1-step argmax), validity at 1 step, q and the
seed-mean one-step weights, for p = (0.6, 0.3, 0.1)."""
import os, sys, json
import numpy as np, torch
from diffusers import AutoencoderKL
from common import load, pilot_shares

torch.set_num_threads(int(os.environ.get("NT", "8")))
HERE = os.path.dirname(os.path.abspath(__file__))
st = load("texture_a90"); fg = st["fg"]; D = st["depth"]; T = 5 * np.sqrt(3)

# measured flatness of the pilot's Marigold outputs (bundled values if the maps are absent)
sh = pilot_shares("mg10*_texture_a90*.npy")
print(f"pilot Marigold outputs (32 maps): object share of the frame's 2/98 range {min(sh):.3f}-{max(sh):.3f} "
      f"(median {np.median(sh):.3f})")

vae = AutoencoderKL.from_pretrained("prs-eth/marigold-depth-v1-0", subfolder="vae", variant="fp16",
                                    torch_dtype=torch.float16).to(torch.float32).eval()
sf = vae.config.scaling_factor


def target(k, gap):
    d = np.where(fg, D[k], np.nanmax(D[:, fg]) + gap)
    a, b = np.percentile(d, [2, 98])
    return np.clip(((d - a) / (b - a) - 0.5) * 2, -1.5, 1.5)


@torch.no_grad()
def enc(d):
    x = torch.tensor(d, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
    return vae.encode(x).latent_dist.mean * sf


src = open(os.path.join(HERE, "check4b_ddim.py"), encoding="utf-8").read().split("PS = [")[0]
ns = {}
exec(src, ns)
run, kappa = ns["run"], ns["kappa"]
geo = {}
for name, gap in [("gap0.5abs", 0.5), ("gap1T", T), ("gap3.2T", 3.2 * T), ("gap7.8T", 7.8 * T), ("gap15T", 15 * T)]:
    tg = np.array([target(k, gap) for k in range(3)])
    Z = torch.cat([enc(tg[k]) for k in range(3)]).reshape(3, -1).double().numpy()
    np.save(os.path.join(HERE, f"latents_{name}.npy"), Z)
    G = Z @ Z.T
    share = np.ptp(tg[0][fg]) / 2
    lo, hi = np.percentile(tg[0][fg], [2, 98]); share98 = (hi - lo) / 2
    geo[name] = dict(gram=G.tolist(), share=float(share), share98=float(share98))
    kap = kappa(G)
    p = np.array([0.6, 0.3, 0.1])
    r1 = run(G, p, 1); r50 = run(G, p, 50)
    print(f"{name:10s} object share of [-1,1]: ptp {share:.3f}, 2-98% {share98:.3f} | |z_k| {np.round(np.sqrt(np.diag(G)),1)} | "
          f"|z_j-z_k| {np.round(kap * (1 - ns['abar'][-1]) / np.sqrt(ns['abar'][-1]), 1)} | kappa {np.round(kap, 2)} | "
          f"iid law, p=(0.6,0.3,0.1): sd(pi_1) {r50['pi_first_sd'][0]:.3f}, P(final = 1-step argmax) {r50['agree_first']:.3f}, "
          f"valid at 1 step {r1['valid']:.3f}, seed-mean 1-step {np.round(r50['mean_pi_first'], 3)}, q {np.round(r50['q'], 3)}",
          flush=True)
json.dump(geo, open(os.path.join(HERE, "vae_geometry_flat.json"), "w"), indent=1)
