"""Surface-normal networks that share a depth network's backbone and training data (Marigold v1.1, Marigold E2E-FT,
Lotus-D), and the exact normals of the stimuli.

A normal map, unlike a depth map, need not integrate around the loop: the three objects and the impossible image have
the same face normals at every shared pixel. Runners return unit normals [H, W, 3] in the network's own axis
convention; scripts/analyze_normals.py fixes the sign of each axis per network on the possible controls.
Ground truth and the analyses use x right, y up, z toward the viewer.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion

from pdepth.models import _device, Lotus
from pdepth.render import _half

PX = 2 * _half(6, 0.12) / 512                    # pixel size of the 512 px stimuli (sets A-C), in cube units


class MarigoldNormals:
    """Marigold normals (v1.1, or E2E-FT run from a zero latent) with the same hand-written DDIM loop as
    pdepth.models.Marigold (trailing spacing, VAE in float32)."""

    def __init__(self, repo_id, variant="fp16", res=768, zero_noise=False):
        from diffusers import MarigoldNormalsPipeline, DDIMScheduler
        self.dev = _device()
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        pipe = MarigoldNormalsPipeline.from_pretrained(repo_id, variant=variant, torch_dtype=torch.float16)
        pipe = pipe.to(dtype=self.dtype).to(self.dev)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.to(torch.float32)
        self.pipe, self.res, self.zero = pipe, res, zero_noise
        with torch.no_grad():
            ids = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(self.dev)
            self.text = pipe.text_encoder(ids)[0].to(self.dtype)

    @torch.no_grad()
    def __call__(self, img_uint8, seed=0, steps=1):
        p = self.pipe
        image, padding, orig = p.image_processor.preprocess(Image.fromarray(img_uint8), self.res, "bilinear",
                                                            self.dev, torch.float32)
        lat = (p.vae.encode(image).latent_dist.mode() * p.vae.config.scaling_factor).to(self.dtype)
        if self.zero:
            x = torch.zeros_like(lat)
        else:
            x = torch.randn(lat.shape, generator=torch.Generator("cpu").manual_seed(int(seed))).to(self.dev, self.dtype)
        p.scheduler.set_timesteps(steps, device=self.dev)
        for t in p.scheduler.timesteps:
            v = p.unet(torch.cat([lat, x], 1), t, encoder_hidden_states=self.text, return_dict=False)[0]
            x = p.scheduler.step(v, t, x).prev_sample
        n = p.decode_prediction(x.float())                                  # [1, 3, h, w], unit length
        n = p.image_processor.unpad_image(n, padding)
        if tuple(n.shape[-2:]) != tuple(orig):
            n = F.interpolate(n, size=orig, mode="bilinear", align_corners=False)
        return _unit(n[0].permute(1, 2, 0).float().cpu().numpy())


class LotusNormal(Lotus):
    """Lotus-D normals: the depth runner's single forward pass, keeping the three decoded channels"""

    @torch.no_grad()
    def __call__(self, img_uint8, seed=0):
        img = torch.from_numpy(img_uint8).float().permute(2, 0, 1)[None] / 255.0
        H, W = img.shape[-2:]
        s = self.res / max(H, W)
        h8, w8 = int(round(H * s / 8) * 8), int(round(W * s / 8) * 8)
        x = F.interpolate(img, (h8, w8), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
        sf = self.vae.config.scaling_factor
        z_x = (self.vae.encode(x.to(self.dev)).latent_dist.mode() * sf).to(self.dt)
        t = torch.tensor([999], device=self.dev)
        z_y = self.unet(z_x, t, encoder_hidden_states=self.emb, class_labels=self.task, return_dict=False)[0]
        y = self.vae.decode(z_y.float() / sf, return_dict=False)[0].clamp(-1, 1)
        y = F.interpolate(y, (H, W), mode="bilinear", align_corners=False)[0]
        return _unit(y.permute(1, 2, 0).cpu().numpy())


def _unit(n):
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


NORMALS = {
    "mg11-n": lambda: MarigoldNormals("prs-eth/marigold-normals-v1-1"),
    "e2eft-n": lambda: MarigoldNormals("GonzaloMartinGarcia/marigold-e2e-ft-normals", variant=None, zero_noise=True),
    "lotus-d-n": lambda: LotusNormal("jingheya/lotus-normal-d-v1-1", "d"),
}


def gt_normals(st, px=PX):
    """exact normals of the shown faces (x right, y up, z toward the viewer) and a mask of face interiors.
    Depth is larger = farther, so n is proportional to (d depth/dx, d depth/dy, 1)."""
    k = int(st["possible"]) if int(st["possible"]) >= 0 else 0
    d = st["depth"][k].astype(float)
    gy, gx = np.gradient(d, px)                                          # per cube unit; rows grow downward
    n = _unit(np.stack([gx, -gy, np.ones_like(d)], -1))
    ok = np.isfinite(n).all(-1) & binary_erosion(st["fg"], iterations=2)
    nn = np.nan_to_num(n)
    for ax in (0, 1):                                                    # drop face edges, where gradients mix
        for sh in (1, -1):
            ok &= (nn * np.roll(nn, sh, ax)).sum(-1) > np.cos(np.deg2rad(2))
    return n, ok


def bar_directions(st):
    """unit direction of travel along each bar (corner j to corner j+1), x right, y up"""
    c = np.asarray(st["corner_xy"], float)
    t = np.stack([c[(j + 1) % 3] - c[j] for j in range(3)])
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def along_bar_slope(n, mask, t):
    """median change of depth per unit length along direction t (depth larger = farther), from normals"""
    v = n[mask]
    return float(np.median((v[:, 0] * t[0] + v[:, 1] * t[1]) / np.maximum(v[:, 2], 1e-3)))
