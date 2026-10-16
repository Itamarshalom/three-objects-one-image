"""Hidden states of a Depth Anything V2 network on stimuli or training scenes, pooled to its 37x37 patch grid, with the
exact per-patch targets, for the layer-wise read-outs of scripts/analyze_layers.py.

Per image: the patch tokens after every ViT block and the four DPT fusion maps (average-pooled to the patch grid), kept
for patches whose pixels are at least 25% figure; the network's own depth output (larger = farther); and per patch the
exact depth gradient g = (d depth/dx, d depth/dy) (x right, y up) averaged over face interiors with its coverage, the
figure's depth (pixel units; stimuli: each of the three objects), the majority cube, and occlusion-edge coverage.
usage: python scripts/run_features.py <dav2-s|dav2-b|dav2-l> <out_dir> <npz files or quoted globs...>
"""
import os, sys, glob, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.normals import gt_normals, PX

REPO_ID = {"dav2-s": "depth-anything/Depth-Anything-V2-Small-hf", "dav2-b": "depth-anything/Depth-Anything-V2-Base-hf",
           "dav2-l": "depth-anything/Depth-Anything-V2-Large-hf"}
G = 37
IDX = np.floor((np.arange(518) + 0.5) * 512 / 518).astype(int)          # 518-px input pixel -> 512-px stimulus pixel


def to_grid(a, w):
    """average a [512, 512, C] map over the pixels where w is true, on the 37x37 grid; returns mean and coverage"""
    a = np.nan_to_num(a[IDX][:, IDX]); w = w[IDX][:, IDX].astype(float)
    if a.ndim == 2:
        a = a[..., None]
    num = (a * w[..., None]).reshape(G, 14, G, 14, -1).sum((1, 3))
    den = w.reshape(G, 14, G, 14).sum((1, 3))
    return num / np.maximum(den, 1e-9)[..., None], den / 196.0


def targets(st):
    """exact per-patch targets; stimuli are told apart from scenes by their 'grad' field"""
    if "grad" in st:                                                          # training scene (pdepth/scenes.py)
        g = np.moveaxis(st["grad"], 0, -1)
        ok = binary_erosion(st["fg"] & ~st["disc"], iterations=2) & np.isfinite(g).all(-1)
        depth = st["depth"][None].astype(float)                              # pixel units
    else:                                                                     # stimulus or ring (cube units -> pixels)
        px = float(st["px"]) if "px" in st else PX
        n, ok = gt_normals(st, px)
        g = np.stack([n[..., 0] / n[..., 2], n[..., 1] / n[..., 2]], -1)
        depth = st["depth"].astype(float) / px
    gg, gcov = to_grid(g, ok)
    dg = np.stack([to_grid(d, st["fg"] & np.isfinite(d))[0][..., 0] for d in depth])
    fcov = to_grid(np.ones((512, 512)), st["fg"])[1]
    disc = to_grid(np.ones((512, 512)), st["disc"])[1] if "disc" in st else np.zeros((G, G))
    cm = st["cube"][IDX][:, IDX].reshape(G, 14, G, 14).transpose(0, 2, 1, 3).reshape(G, G, -1)
    maj = np.full((G, G), -1)
    for r in range(G):
        for c in range(G):
            v = cm[r, c][cm[r, c] >= 0]
            if len(v):
                maj[r, c] = np.bincount(v).argmax()
    return gg, gcov, dg, fcov, disc, maj


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    key, out = sys.argv[1], sys.argv[2]
    files = sorted(f for a in sys.argv[3:] for f in (glob.glob(a) if any(c in a for c in "*?[") else [a]))
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    proc = AutoImageProcessor.from_pretrained(REPO_ID[key])
    model = AutoModelForDepthEstimation.from_pretrained(REPO_ID[key]).eval().to(dev)
    cap = {}
    model.neck.fusion_stage.register_forward_hook(lambda m, i, o: cap.__setitem__("fus", o))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for i, f in enumerate(files):
        name = os.path.basename(f)[:-4]
        if os.path.exists(os.path.join(out, name + ".npz")):
            continue
        st = dict(np.load(f))
        inp = {k: v.to(dev) for k, v in proc(images=Image.fromarray(st["img"]), return_tensors="pt").items()}
        assert inp["pixel_values"].shape[-1] == 518, "expects square 512 px images"
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True)
        gg, gcov, dg, fcov, disc, maj = targets(st)
        keep = (fcov >= 0.25).reshape(-1)
        hs = torch.stack([h[0, 1:] for h in o.hidden_states]).float().cpu().numpy()[:, keep]
        fus = np.stack([F.adaptive_avg_pool2d(x, G)[0].permute(1, 2, 0).reshape(G * G, -1).float().cpu().numpy()[keep]
                        for x in cap["fus"]]) if len({x.shape[1] for x in cap["fus"]}) == 1 else None
        pd = o.predicted_depth[:, None].float()
        dep = -F.interpolate(pd, size=(512, 512), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        np.savez(os.path.join(out, name + ".npz"), keep=keep, hs=hs.astype(np.float16),
                 fus=(fus.astype(np.float16) if fus is not None else np.zeros(0)), dep=dep.astype(np.float32),
                 dep_grid=to_grid(dep, st["fg"])[0][..., 0].astype(np.float32), grad=gg.astype(np.float32),
                 gcov=gcov.astype(np.float32), dgrid=dg.astype(np.float32), fcov=fcov.astype(np.float32),
                 disc=disc.astype(np.float32), maj=maj.astype(np.int16),
                 **{k: st[k] for k in ("set", "angle", "mirror", "possible", "corner_xy", "n_bars") if k in st})
        if (i + 1) % 25 == 0:
            print(f"{key}: {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done {key}: {len(files)} files ({time.time() - t0:.0f}s)")
