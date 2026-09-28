"""Depth model runners. Every runner returns depth maps at the stimulus resolution with the
convention larger = farther (up to an unknown scale and shift, which the read-out removes)."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class HFDepth:
    """transformers depth models (Depth Anything V2, DPT/MiDaS, ZoeDepth, ...).
    kind: 'disparity' if the model outputs relative inverse depth (larger = nearer), else 'depth'."""

    def __init__(self, repo_id, kind="disparity"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.dev = _device()
        self.proc = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo_id).eval().to(self.dev)
        self.kind = kind

    @torch.no_grad()
    def __call__(self, img_uint8, seed=0):
        img = Image.fromarray(img_uint8)
        inp = {k: v.to(self.dev) for k, v in self.proc(images=img, return_tensors="pt").items()}
        outputs = self.model(**inp)
        if type(self.proc).__name__.startswith("ZoeDepth"):
            # ZoeDepth's processor reflect-pads the image, so the padding must be cropped before the map lines up
            out = self.proc.post_process_depth_estimation(outputs, source_sizes=[img.size[::-1]])[0]["predicted_depth"]
            out = out.float().cpu().numpy()
        else:
            out = outputs.predicted_depth
            out = F.interpolate(out[:, None] if out.dim() == 3 else out, size=img.size[::-1], mode="bicubic",
                                align_corners=False)[0, 0].float().cpu().numpy()
        return -out if self.kind == "disparity" else out


class Marigold:
    """Marigold-family models (Marigold v1-0 / v1-1, E2E-FT) with a hand-written DDIM loop, the same
    computation as diffusers' MarigoldDepthPipeline with ensemble_size=1 and the same scheduler (checked to
    2e-5 on v1-1), so that several seeds run as one batch and the x0-prediction at every step can be decoded.
    v1-0 ships with leading timestep spacing (a single step at t=1), so every checkpoint is switched to
    trailing spacing below (Martin Garcia et al. 2025)."""

    def __init__(self, repo_id="prs-eth/marigold-depth-v1-1", variant="fp16", res=768, spacing="trailing"):
        from diffusers import MarigoldDepthPipeline, DDIMScheduler
        self.dev = _device()
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        pipe = MarigoldDepthPipeline.from_pretrained(repo_id, variant=variant, torch_dtype=torch.float16)
        pipe = pipe.to(dtype=self.dtype).to(self.dev)
        # trailing spacing: the first step starts at the last training timestep, also for 1 step
        # (keeps each checkpoint's own beta schedule, incl. v1-1's zero terminal SNR)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing=spacing)
        pipe.set_progress_bar_config(disable=True)
        # the VAE runs in float32: the object's depth range is small and fp16 would quantise it
        pipe.vae.to(torch.float32)
        self.pipe, self.res = pipe, res
        with torch.no_grad():
            ids = pipe.tokenizer("", padding="do_not_pad", max_length=pipe.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(self.dev)
            self.text = pipe.text_encoder(ids)[0].to(self.dtype)

    @torch.no_grad()
    def encode(self, img_uint8):
        p = self.pipe
        image, padding, orig = p.image_processor.preprocess(Image.fromarray(img_uint8), self.res, "bilinear",
                                                            self.dev, torch.float32)
        lat = p.vae.encode(image).latent_dist.mode() * p.vae.config.scaling_factor
        return lat.to(self.dtype), padding, orig

    @torch.no_grad()
    def decode(self, latent, padding, orig, chunk=4):
        p = self.pipe
        # float32 decoding in chunks: a batch of 16 at 768 px would not fit in 24 GB
        pred = torch.cat([p.decode_prediction(latent[i:i + chunk].float()) for i in range(0, len(latent), chunk)])
        pred = p.image_processor.unpad_image(pred, padding)
        if tuple(pred.shape[-2:]) != tuple(orig):
            pred = F.interpolate(pred, size=orig, mode="bilinear", align_corners=False)
        return pred[:, 0].float().cpu().numpy()

    def noise(self, shape, seeds, zero=False):
        if zero:
            return torch.zeros((len(seeds),) + tuple(shape), device=self.dev, dtype=self.dtype)
        return torch.stack([torch.randn(shape, generator=torch.Generator("cpu").manual_seed(int(s)))
                            for s in seeds]).to(self.dev, self.dtype)

    @torch.no_grad()
    def sample(self, img_uint8, seeds, steps, record_x0=False, batch=16, zero_noise=False, init=None):
        """final depths [S,H,W]; with record_x0 also x0-predictions [S,steps,H,W].
        init: optional explicit starting latents [S,4,h,w] (for noise-swap interventions)."""
        p = self.pipe
        img_lat, padding, orig = self.encode(img_uint8)
        finals, trajs = [], []
        for b0 in range(0, len(seeds), batch):
            sb = seeds[b0:b0 + batch]
            x = self.noise(img_lat.shape[1:], sb, zero_noise) if init is None else init[b0:b0 + batch].to(self.dev, self.dtype)
            lat_img = img_lat.expand(len(sb), -1, -1, -1)
            text = self.text.expand(len(sb), -1, -1)
            p.scheduler.set_timesteps(steps, device=self.dev)
            traj = []
            for t in p.scheduler.timesteps:
                v = p.unet(torch.cat([lat_img, x], 1), t, encoder_hidden_states=text, return_dict=False)[0]
                out = p.scheduler.step(v, t, x)
                x = out.prev_sample
                if record_x0:
                    traj.append(self.decode(out.pred_original_sample, padding, orig))
            finals.append(self.decode(x, padding, orig))
            if record_x0:
                trajs.append(np.stack(traj, 1))
        finals = np.concatenate(finals)
        return (finals, np.concatenate(trajs)) if record_x0 else finals

    def __call__(self, img_uint8, seed=0, steps=1, zero_noise=False):
        return self.sample(img_uint8, [seed], steps, zero_noise=zero_noise)[0]


class Lotus:
    """Lotus-D / Lotus-G one-step forward (EnVision-Research/Lotus@b737e12 pipeline.py, re-implemented
    for current diffusers). x0-prediction at t=999; D sees the image latent only, G also a noise latent.
    The image is encoded with the latent mode (the official code samples, making even Lotus-D random)."""

    def __init__(self, repo_id, mode, res=768):
        from diffusers import UNet2DConditionModel, AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer
        self.dev = _device()
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet").to(self.dev, self.dt).eval()
        self.vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae").to(self.dev, torch.float32).eval()
        tok = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder").to(self.dev).eval()
        with torch.no_grad():
            ids = tok("", padding="do_not_pad", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
            self.emb = te(ids.to(self.dev))[0].to(self.dt)
        task = torch.tensor([[1.0, 0.0]], device=self.dev)                  # [1,0] = annotation branch
        self.task = torch.cat([torch.sin(task), torch.cos(task)], -1).to(self.dt)
        self.mode, self.res = mode, res
        self.disparity = "disparity" in repo_id

    @torch.no_grad()
    def __call__(self, img_uint8, seed=0):
        img = torch.from_numpy(img_uint8).float().permute(2, 0, 1)[None] / 255.0
        H, W = img.shape[-2:]
        s = self.res / max(H, W)
        h8, w8 = int(round(H * s / 8) * 8), int(round(W * s / 8) * 8)
        x = F.interpolate(img, (h8, w8), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
        sf = self.vae.config.scaling_factor
        z_x = (self.vae.encode(x.to(self.dev)).latent_dist.mode() * sf).to(self.dt)
        if self.mode == "g":
            eps = torch.randn(z_x.shape, generator=torch.Generator("cpu").manual_seed(int(seed))).to(self.dev, self.dt)
            inp = torch.cat([z_x, eps], 1)
        else:
            inp = z_x
        t = torch.tensor([999], device=self.dev)
        z_y = self.unet(inp, t, encoder_hidden_states=self.emb, class_labels=self.task, return_dict=False)[0]
        y = self.vae.decode(z_y.float() / sf, return_dict=False)[0].mean(1, keepdim=True)
        y = F.interpolate(y, (H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        return -y if self.disparity else y


REGISTRY = {
    # L1-type deterministic (fixed gauge: median / MAD normalisation, MAE)
    "dav2-s": lambda: HFDepth("depth-anything/Depth-Anything-V2-Small-hf"),
    "dav2-b": lambda: HFDepth("depth-anything/Depth-Anything-V2-Base-hf"),
    "dav2-l": lambda: HFDepth("depth-anything/Depth-Anything-V2-Large-hf"),
    "dpt-l": lambda: HFDepth("Intel/dpt-large"),
    # log-L2 (SILog) metric depth
    "zoe": lambda: HFDepth("Intel/zoedepth-nyu-kitti", kind="depth"),
    # SD2 family, one step
    "lotus-d": lambda: Lotus("jingheya/lotus-depth-d-v2-0-disparity", "d"),
    "lotus-g": lambda: Lotus("jingheya/lotus-depth-g-v2-1-disparity", "g"),
    "e2eft": lambda: _ZeroNoise(Marigold("GonzaloMG/marigold-e2e-ft-depth", variant=None)),
    "mg11-1": lambda: Marigold("prs-eth/marigold-depth-v1-1"),
    "mg10-1": lambda: Marigold("prs-eth/marigold-depth-v1-0"),
}


class _ZeroNoise:
    """E2E-FT is trained and run from a zero latent in a single step"""

    def __init__(self, m):
        self.m = m

    def __call__(self, img_uint8, seed=0):
        return self.m(img_uint8, seed, steps=1, zero_noise=True)
