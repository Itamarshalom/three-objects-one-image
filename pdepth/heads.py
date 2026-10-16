"""Matched read-out heads on a frozen Depth Anything V2 backbone: a 0-form head that predicts depth and a 1-form head
that predicts the depth gradient, identical except for the output.

Both heads are the network's own DPT decoder (neck + head, built from the transformers config with random init, or with
the pretrained neck) on the frozen backbone's feature maps at the DPT tap layers (S: blocks 3, 6, 9, 12), with the
final ReLU removed. Outputs at the input resolution:
- 0-form head: depth (1 channel, pixel units, larger = farther; predicted in units of depth_unit px) + edge logit;
- 1-form head: g = (d depth/dx, d depth/dy) (2 channels, x right, y up, dimensionless, as in pdepth/scenes.py)
  + edge logit.
Losses (scripts/train_heads.py, on the 512 px labels of data/scenes; the edge target is the scenes' occlusion edges):
- 0-form: shift-invariant L1 on depth (the shift is the median residual over the figure, the L1-optimal one) +
  multi-scale gradient matching of the residual (MiDaS; each scale's differences are divided by their pixel step, so
  the finest scale is the 1-form loss of the depth's finite differences) + edge BCE;
- 1-form: L1 on g over the figure outside a 2 px band around occlusion edges + edge BCE.
HeadRunner loads a checkpoint (outputs/heads/<name>.pt, relative to the repo root) and runs backbone and head on an
image; a 0-form head is the REGISTRY key head-<name> of pdepth/models.py. Its edge map is a calibrated probability:
the BCE's pos_weight w trains the logit to log w + logit(p), so p = sigmoid(logit - log w) (on held-out scenes the raw
sigmoid(logit) read 0.56 where the edge frequency was 0.14).
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

REPO_ID = {"dav2-s": "depth-anything/Depth-Anything-V2-Small-hf", "dav2-b": "depth-anything/Depth-Anything-V2-Base-hf",
           "dav2-l": "depth-anything/Depth-Anything-V2-Large-hf"}
N_OUT = {"0form": 1, "1form": 2}                    # output channels before the edge logit
HEAD_DIR = "outputs/heads"                          # checkpoints, relative to the repo root like every other output
BAND = 2                                            # px: the 1-form loss skips this band around occlusion edges
SCALES = 4                                          # gradient matching at steps 1, 2, 4, 8 px
# label bit masks (scene_labels)
FG, EDGE, GOK, INNER = 1, 2, 4, 8


def no_tf32():
    """true float32 on Ampere GPUs (the head runs in float32): TF32 keeps float16's 10-bit mantissa, which changed the
    1 px slopes of the pretrained DAv2-S output by 1.4-2.7% (RMS) in a CPU emulation (a float16 neck: 0.2-0.35%)"""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_dav2(key):
    """image processor and pretrained network, frozen and in eval mode"""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(REPO_ID[key])
    model = AutoModelForDepthEstimation.from_pretrained(REPO_ID[key]).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def pixel_values(proc, img_uint8):
    """the network's own preprocessing (512 px -> 518 px, ImageNet normalisation): [1, 3, 518, 518]"""
    return proc(images=Image.fromarray(img_uint8), return_tensors="pt")["pixel_values"]


@torch.no_grad()
def backbone_tokens(backbone, pv):
    """patch tokens at the DPT tap layers, layer-normed as the neck gets them, class token dropped: [B, 4, h*w, C]"""
    return torch.stack([f[:, 1:] for f in backbone(pixel_values=pv).feature_maps], 1)


class Decoder(nn.Module):
    """the DAv2 DPT decoder: neck (reassemble + fusion) and head (3x3 conv, upsampling to the input size, 3x3 conv,
    ReLU, 1x1 conv) with n_out + 1 output channels (the last is the edge logit) and no final activation"""

    def __init__(self, config, n_out):
        super().__init__()
        from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingNeck
        self.neck = DepthAnythingNeck(config)
        f, h = config.fusion_hidden_size, config.head_hidden_size
        # same layer names as the pretrained head, so its conv1/conv2 can be loaded
        self.head = nn.ModuleDict(dict(conv1=nn.Conv2d(f, f // 2, 3, padding=1),
                                       conv2=nn.Conv2d(f // 2, h, 3, padding=1), conv3=nn.Conv2d(h, n_out + 1, 1)))
        self.patch = config.patch_size

    def forward(self, tok, h, w, amp=False):
        """tok [B, 4, h*w, C] -> [B, n_out + 1, 14h, 14w]. With amp the neck runs in float16 on CUDA; the head always
        runs in float32 (depth in pixel units needs the precision for pixel-scale slopes)"""
        tok = tok.float()
        cls = tok.new_zeros(tok.shape[0], 1, tok.shape[-1])                  # the neck drops the class token anyway
        with torch.autocast(tok.device.type, dtype=torch.float16, enabled=amp):
            x = self.neck([torch.cat([cls, tok[:, i]], 1) for i in range(tok.shape[1])], h, w)[-1]
        x = self.head["conv1"](x.float())
        x = F.interpolate(x, (h * self.patch, w * self.patch), mode="bilinear", align_corners=True)
        return self.head["conv3"](F.relu(self.head["conv2"](x)))


def init_from(dec, model, how):
    """'random' (PyTorch default init), 'neck' (the pretrained neck) or 'neck+head' (also the head's conv1 and conv2)"""
    if how not in ("random", "neck", "neck+head"):
        raise ValueError(f"unknown init {how}")
    if how != "random":
        dec.neck.load_state_dict(model.neck.state_dict())
    if how == "neck+head":
        for k in ("conv1", "conv2"):
            dec.head[k].load_state_dict(getattr(model.head, k).state_dict())


def scene_labels(st):
    """labels of one training scene (pdepth/scenes.py) at 512 px: depth (pixel units, 0 off the figure), g [2, H, W]
    (0 off the figure) and bit masks FG figure, EDGE occlusion edge, GOK g supervised (figure outside the edge band),
    INNER g supervised with the same cube face on all 4 neighbours (where the two heads' slopes are compared)"""
    fg, disc = st["fg"], st["disc"]
    ok = fg & np.isfinite(st["grad"]).all(0) & ~binary_dilation(disc, iterations=BAND)
    lab = np.where(fg, st["cube"].astype(int) * 3 + st["face"], -1)
    inner = ok.copy()
    for ax in (0, 1):
        for sh in (1, -1):
            inner &= lab == np.roll(lab, sh, ax)
    inner[[0, -1]] = False; inner[:, [0, -1]] = False
    bits = fg * FG + disc * EDGE + ok * GOK + inner * INNER
    return (np.nan_to_num(st["depth"]).astype(np.float32), np.nan_to_num(st["grad"]).astype(np.float16),
            bits.astype(np.uint8))


def implied_g(depth):
    """g of depth maps [..., H, W] (pixel units) by central differences, x right, y up (rows grow downward)"""
    gx = F.pad((depth[..., :, 2:] - depth[..., :, :-2]) / 2, (1, 1))
    gy = -F.pad((depth[..., 2:, :] - depth[..., :-2, :]) / 2, (0, 0, 1, 1))
    return torch.stack([gx, gy], -3)


def _grad_match(r, m):
    """mean |x difference| + mean |y difference| of the residual r over pixel pairs inside the mask"""
    mx, my = m[:, :, 1:] & m[:, :, :-1], m[:, 1:] & m[:, :-1]
    gx = ((r[:, :, 1:] - r[:, :, :-1]).abs() * mx).sum() / mx.sum().clamp(min=1)
    gy = ((r[:, 1:] - r[:, :-1]).abs() * my).sum() / my.sum().clamp(min=1)
    return gx + gy


def head_loss(kind, out, depth, g, bits, depth_unit=64.0, w_grad=1.0, w_edge=1.0, pos_weight=10.0):
    """loss of a batch: out [B, n_out + 1, H, W] at the label resolution, depth [B, H, W] (px), g [B, 2, H, W],
    bits [B, H, W] (scene_labels). Pixel terms are pooled over the batch. Returns the loss and its parts (tensors)"""
    edge = F.binary_cross_entropy_with_logits(out[:, -1], ((bits & EDGE) > 0).float(),
                                              pos_weight=torch.tensor(pos_weight, device=out.device))
    if kind == "1form":
        ok = (bits & GOK) > 0
        l1 = ((out[:, :2] - g).abs().sum(1) * ok).sum() / ok.sum().clamp(min=1)
        return l1 + w_edge * edge, dict(l1=l1, edge=edge)
    fg = (bits & FG) > 0
    r = torch.where(fg, out[:, 0] * depth_unit - depth, torch.zeros_like(depth))             # residual, px
    shift = torch.nan_to_num(torch.nanmedian(torch.where(fg, r, torch.nan).flatten(1), 1).values.detach())
    r = torch.where(fg, r - shift[:, None, None], torch.zeros_like(r))
    l1 = r.abs().sum() / fg.sum().clamp(min=1) / depth_unit
    gm = sum(_grad_match(r[:, ::s, ::s], fg[:, ::s, ::s]) / s for s in [2 ** k for k in range(SCALES)]) / SCALES
    return l1 + w_grad * gm + w_edge * edge, dict(l1=l1, gm=gm, edge=edge)


@torch.no_grad()
def head_metrics(kind, out, depth, g, bits, depth_unit=64.0):
    """sums for the validation metrics that compare the two heads: slope error |g_hat - g|_1 on face interiors (0-form:
    central differences of its depth), edge counts at logit 0, i.e. calibrated p = 1/(1 + pos_weight) (0.09 by default;
    at p = 0.5 the F1 of 64-step test heads was still 0), and the 0-form's shift-invariant depth error (px)"""
    inner = (bits & INNER) > 0
    gh = implied_g(out[:, 0] * depth_unit) if kind == "0form" else out[:, :2]
    p, e = out[:, -1] > 0, (bits & EDGE) > 0
    s = dict(slope_sum=((gh - g).abs().sum(1) * inner).sum(), slope_n=inner.sum(), tp=(p & e).sum(), fp=(p & ~e).sum(),
             fn=(~p & e).sum())
    if kind == "0form":
        fg = (bits & FG) > 0
        r = torch.where(fg, out[:, 0] * depth_unit - depth, torch.nan).flatten(1)
        s.update(mae_sum=(r - r.nanmedian(1, keepdim=True).values).abs().nansum(), mae_n=fg.sum())
    return {k: float(v) for k, v in s.items()}


def ckpt_path(name):
    return name if name.endswith(".pt") else os.path.join(HEAD_DIR, name + ".pt")


class HeadRunner:
    """a trained head on its frozen backbone. __call__(img) returns depth [H, W] (0-form; pixel units of the image,
    larger = farther) or (g [2, H, W], edge probability [H, W]) (1-form); maps(img) returns all maps as a dict.
    The edge probability is calibrated: sigmoid(logit - log pos_weight) (module docstring)."""

    def __init__(self, ckpt, kind=None, device=None):
        ck = torch.load(ckpt_path(ckpt), map_location="cpu", weights_only=True)
        if kind is not None and ck["kind"] != kind:
            raise ValueError(f"{ckpt} is a {ck['kind']} head, not {kind}")
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        no_tf32()
        self.proc, model = load_dav2(ck["backbone"])
        self.backbone = model.backbone.to(self.dev)
        self.dec = Decoder(model.config, N_OUT[ck["kind"]])
        self.dec.load_state_dict(ck["decoder"])
        self.dec.eval().to(self.dev)
        self.kind, self.unit, self.patch = ck["kind"], float(ck["depth_unit"]), model.config.patch_size
        self.log_w = math.log(float(ck.get("args", {}).get("pos_weight", 1.0)))     # prior shift of the edge logit

    @torch.no_grad()
    def maps(self, img_uint8):
        pv = pixel_values(self.proc, img_uint8).to(self.dev)
        h, w = pv.shape[-2] // self.patch, pv.shape[-1] // self.patch
        tok = backbone_tokens(self.backbone, pv).half()                       # rounded like the training cache
        out = F.interpolate(self.dec(tok, h, w), img_uint8.shape[:2], mode="bilinear", align_corners=False)[0]
        edge = torch.sigmoid(out[-1] - self.log_w).cpu().numpy()
        if self.kind == "0form":
            return dict(depth=(out[0] * self.unit).cpu().numpy(), edge=edge)
        return dict(g=out[:2].cpu().numpy(), edge=edge)

    def __call__(self, img_uint8, seed=0):
        m = self.maps(img_uint8)
        return m["depth"] if self.kind == "0form" else (m["g"], m["edge"])
