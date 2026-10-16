"""Train one matched read-out head (pdepth/heads.py) on the frozen Depth Anything V2 backbone with the POSSIBLE training
scenes of data/scenes: kind 0form (depth + edge logit) or 1form (depth gradient g + edge logit).

The backbone's tokens at the DPT tap layers are computed once per scene and cached as float16, one .npy per scene in
<cache>/<backbone>/ (shared by both heads and all seeds; recomputed when the scene file is newer, so regenerated scenes
never meet stale tokens), then held in RAM (--disk reads each batch from the cache instead).
The validation scenes are the last --n-val scenes in name order, the same for every head and seed. AdamW, linear
warm-up then cosine decay, gradient clipping, AMP on CUDA (float16 neck; the head in float32 with TF32 off). Every
epoch ends with a validation pass (also before training) and rewrites outputs/heads/<name>.pt (decoder, settings,
history, optimizer state; --resume continues from it) and outputs/heads/<name>.csv (one row per validation). Validation
metrics, the same for both heads: loss, slope error |g_hat - g|_1 on face interiors (0-form: central differences of its
depth), edge F1 at logit 0 (calibrated p = 1/(1 + pos_weight); see pdepth/heads.py), and for the 0-form the
shift-invariant depth error in px.
usage: python scripts/train_heads.py <0form|1form> <name> [options]     (options: python scripts/train_heads.py -h)
GPU benchmark: python scripts/train_heads.py 0form bench --max-steps 60 --log-every 20     (s/step of the last lines)
"""
import os, sys, glob, csv, time, math, argparse, threading, queue
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.heads import (REPO_ID, N_OUT, HEAD_DIR, no_tf32, load_dav2, pixel_values, backbone_tokens, Decoder,
                          init_from, scene_labels, head_loss, head_metrics)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("kind", choices=sorted(N_OUT))
    p.add_argument("name", help="checkpoint name: outputs/heads/<name>.pt")
    p.add_argument("--scenes", default="data/scenes/scene_*.npz", help="training scenes (quoted glob)")
    p.add_argument("--n-scenes", type=int, default=0, help="use only the first N scenes in name order (0 = all)")
    p.add_argument("--n-val", type=int, default=100, help="validation scenes (the last ones in name order)")
    p.add_argument("--backbone", default="dav2-s", choices=sorted(REPO_ID))
    p.add_argument("--init", default="random", choices=["random", "neck", "neck+head"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2, help="AdamW weight decay")
    p.add_argument("--warmup", type=float, default=0.03, help="warm-up, fraction of all steps")
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-unit", type=float, default=64.0, help="0-form: output and L1 unit, px (about one cube)")
    p.add_argument("--w-grad", type=float, default=1.0, help="0-form: weight of the gradient matching")
    p.add_argument("--w-edge", type=float, default=1.0, help="weight of the edge BCE")
    p.add_argument("--pos-weight", type=float, default=10.0, help="edge BCE weight of edge pixels")
    p.add_argument("--cache", default=os.path.join(HEAD_DIR, "cache"), help="token cache root")
    p.add_argument("--disk", action="store_true", help="read tokens from the cache on disk instead of RAM")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (benchmarks, smoke tests)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    p.add_argument("--workers", type=int, default=8, help="threads that load the labels")
    p.add_argument("--resume", action="store_true", help="continue from outputs/heads/<name>.pt")
    return p.parse_args()


def tok_path(cache, f):
    return os.path.join(cache, os.path.basename(f)[:-4] + ".npy")


def _img(f):
    with np.load(f) as z:
        return z["img"]


def cache_tokens(files, proc, backbone, cache, dev, bs=8):
    """frozen-backbone tokens [4, h*w, C] of every scene not cached yet or newer than its tokens (float16, written
    atomically)"""
    todo = [f for f in files if not os.path.exists(tok_path(cache, f)) or
            os.path.getmtime(tok_path(cache, f)) < os.path.getmtime(f)]
    if not todo:
        return
    stale = sum(os.path.exists(tok_path(cache, f)) for f in todo)
    if stale:
        print(f"{stale} scenes are newer than their cached tokens: recomputing them", flush=True)
    os.makedirs(cache, exist_ok=True)
    backbone.to(dev)
    t0 = time.time()
    for i in range(0, len(todo), bs):
        fs = todo[i:i + bs]
        pv = torch.cat([pixel_values(proc, _img(f)) for f in fs]).to(dev)
        tok = backbone_tokens(backbone, pv).half().cpu().numpy()
        for f, t in zip(fs, tok):
            with open(tok_path(cache, f) + ".tmp", "wb") as fh:
                np.save(fh, t)
            os.replace(tok_path(cache, f) + ".tmp", tok_path(cache, f))
        if (i // bs + 1) % 25 == 0 or i + bs >= len(todo):
            print(f"cached tokens {min(i + bs, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s, "
                  f"{(time.time() - t0) / min(i + bs, len(todo)):.3f} s/scene)", flush=True)
    backbone.to("cpu")


def load_tokens(files, cache, disk):
    """all tokens in RAM [N, 4, h*w, C], or with disk the cache paths, read per batch (a memmap per scene would keep
    one file handle open per scene, past Linux's usual limit of 1024 at 1500 scenes); and the shape of one scene"""
    shape = np.load(tok_path(cache, files[0]), mmap_mode="r").shape
    if disk:
        return [tok_path(cache, f) for f in files], shape
    T = np.empty((len(files),) + shape, np.float16)
    for i, f in enumerate(files):
        T[i] = np.load(tok_path(cache, f))
    return T, shape


def load_labels(files, workers):
    """depth, g, bit masks of every scene (pdepth.heads.scene_labels), in RAM"""
    with np.load(files[0]) as z:
        H, W = z["fg"].shape
    D, G, B = np.empty((len(files), H, W), np.float32), np.empty((len(files), 2, H, W), np.float16), \
        np.empty((len(files), H, W), np.uint8)

    def fill(i):
        with np.load(files[i]) as z:
            D[i], G[i], B[i] = scene_labels({k: z[k] for k in ("depth", "grad", "disc", "fg", "cube", "face")})

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fill, range(len(files))))
    return D, G, B


def batches(idx, bs, get, prefetch=3):
    """batches of the given scene indices, assembled in a background thread"""
    q = queue.Queue(prefetch)

    def work():
        try:
            for i in range(0, len(idx), bs):
                q.put(get(np.sort(idx[i:i + bs])))
        except Exception as e:                                                   # re-raised in the main thread
            q.put(e)
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (b := q.get()) is not None:
        if isinstance(b, Exception):
            raise b
        yield b


def lr_factor(step, warm, total):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def main():
    a = parse()
    torch.set_num_threads(a.threads or os.cpu_count())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda" and not a.no_amp
    no_tf32()
    files = sorted(glob.glob(a.scenes))[:a.n_scenes or None]
    if a.n_val < 1 or len(files) <= a.n_val:
        sys.exit(f"{len(files)} scenes found for {a.scenes}; need --n-val >= 1 and more scenes than that")
    tr, va = np.arange(len(files) - a.n_val), np.arange(len(files) - a.n_val, len(files))
    os.makedirs(HEAD_DIR, exist_ok=True)
    out_pt, out_csv = os.path.join(HEAD_DIR, a.name + ".pt"), os.path.join(HEAD_DIR, a.name + ".csv")
    print(f"{a.name}: {a.kind} head on {a.backbone}, init {a.init}, {len(tr)} train / {len(va)} val scenes, "
          f"device {dev}, amp {amp}, torch threads {torch.get_num_threads()}", flush=True)

    # 1. tokens of the frozen backbone (cached once per scene) and labels, in RAM
    proc, model = load_dav2(a.backbone)
    cache = os.path.join(a.cache, a.backbone)
    cache_tokens(files, proc, model.backbone, cache, dev)
    t0 = time.time()
    T, shape = load_tokens(files, cache, a.disk)
    D, G, B = load_labels(files, a.workers)
    n_tok = shape[1]
    h = w = math.isqrt(n_tok)
    assert h * w == n_tok, "square images only"
    print(f"data in {time.time() - t0:.0f}s: tokens {'on disk' if a.disk else f'{T.nbytes / 1e9:.2f} GB'}, "
          f"labels {(D.nbytes + G.nbytes + B.nbytes) / 1e9:.2f} GB, patch grid {h}x{w}", flush=True)
    pin = dev == "cuda"

    def get(ix):
        tok = T[ix] if not a.disk else np.stack([np.load(T[i]) for i in ix])
        b = [torch.from_numpy(np.ascontiguousarray(x)) for x in (tok, D[ix], G[ix], B[ix])]
        return [x.pin_memory() for x in b] if pin else b

    # 2. decoder, optimiser, schedule
    torch.manual_seed(a.seed)
    dec = Decoder(model.config, N_OUT[a.kind])
    init_from(dec, model, a.init)
    del model
    dec.to(dev)
    spe = len(tr) // a.batch
    if spe == 0:
        sys.exit(f"--batch {a.batch} is larger than the {len(tr)} training scenes")
    total = a.epochs * spe
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, max(1, round(a.warmup * total)), total))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    kw = dict(depth_unit=a.depth_unit, w_grad=a.w_grad, w_edge=a.w_edge, pos_weight=a.pos_weight)
    print(f"decoder {sum(p.numel() for p in dec.parameters()) / 1e6:.2f} M parameters, {spe} steps/epoch, "
          f"{total} steps", flush=True)
    epoch, step, hist = 0, 0, []
    if a.resume and os.path.exists(out_pt):
        ck = torch.load(out_pt, map_location="cpu", weights_only=True)
        if ck["kind"] != a.kind or ck["backbone"] != a.backbone:
            sys.exit(f"{out_pt} is a {ck['kind']} head on {ck['backbone']}")
        if ck["val_scenes"] != [os.path.basename(files[i])[:-4] for i in va]:
            sys.exit(f"{out_pt} was validated on other scenes: resume with its --scenes, --n-scenes and --n-val")
        dec.load_state_dict(ck["decoder"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        epoch, step, hist = ck["epoch"], ck["step"], ck["history"]
        print(f"resumed {out_pt} at epoch {epoch}, step {step}" + (" (saved mid-epoch: that epoch restarts)"
                                                                     if ck.get("partial") else ""), flush=True)

    def to_dev(b):
        tok, depth, g, bits = (x.to(dev, non_blocking=True) for x in b)
        return tok, depth, g.float(), bits

    def forward(tok, size):
        return F.interpolate(dec(tok, h, w, amp), size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def validate():
        dec.eval()
        s = defaultdict(float)
        for b in batches(va, a.batch, get):
            tok, depth, g, bits = to_dev(b)
            out = forward(tok, depth.shape[-2:])
            loss, parts = head_loss(a.kind, out, depth, g, bits, **kw)
            n = len(tok)
            s["n"] += n; s["loss"] += float(loss) * n
            for k, v in parts.items():
                s[k] += float(v) * n
            for k, v in head_metrics(a.kind, out, depth, g, bits, a.depth_unit).items():
                s[k] += v
        dec.train()
        r = {f"val_{k}": s[k] / s["n"] for k in ["loss"] + list(parts)}
        r["val_slope_err"] = s["slope_sum"] / max(s["slope_n"], 1)
        r["val_edge_f1"] = 2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1)
        if a.kind == "0form":
            r["val_depth_mae_px"] = s["mae_sum"] / max(s["mae_n"], 1)
        return r

    def save(partial=False):
        ck = dict(kind=a.kind, backbone=a.backbone, init=a.init, depth_unit=a.depth_unit, decoder=dec.state_dict(),
                  epoch=epoch, step=step, partial=partial, history=hist, args=vars(a), n_train=len(tr),
                  val_scenes=[os.path.basename(files[i])[:-4] for i in va], opt=opt.state_dict(),
                  sched=sched.state_dict(), scaler=scaler.state_dict(), torch_version=str(torch.__version__))
        torch.save(ck, out_pt + ".tmp")
        os.replace(out_pt + ".tmp", out_pt)
        with open(out_csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in hist for k in r)), restval="")
            wr.writeheader(); wr.writerows(hist)

    def report(row):
        hist.append(row)
        print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

    # 3. training
    t_start = time.time()
    if not hist:
        report(dict(epoch=0, step=0, **validate()))
    stop = False
    while epoch < a.epochs and not stop:
        order = tr[np.random.default_rng(a.seed * 100003 + epoch).permutation(len(tr))[:spe * a.batch]]
        run, n_run, t_ep, t_log, n_ep = defaultdict(float), 0, time.time(), time.time(), 0
        for b in batches(order, a.batch, get):
            tok, depth, g, bits = to_dev(b)
            out = forward(tok, depth.shape[-2:])
            loss, parts = head_loss(a.kind, out, depth, g, bits, **kw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(dec.parameters(), a.clip)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; n_ep += 1; n_run += 1
            run["loss"] += loss.detach()
            for k, v in parts.items():
                run[k] += v.detach()
            if step % a.log_every == 0 or (a.max_steps and step >= a.max_steps):
                vals = {k: float(v) / n_run for k, v in run.items()}
                if not all(map(math.isfinite, vals.values())):
                    sys.exit(f"non-finite training loss at step {step}: {vals}")
                if dev == "cuda":
                    torch.cuda.synchronize()
                print(f"epoch {epoch + 1} step {step} | " + " ".join(f"{k} {v:.4f}" for k, v in vals.items()) +
                      f" | lr {sched.get_last_lr()[0]:.2e} | {(time.time() - t_log) / n_run:.3f} s/step", flush=True)
                run, n_run, t_log = defaultdict(float), 0, time.time()
            if a.max_steps and step >= a.max_steps:
                stop = True
                break
        if dev == "cuda":
            torch.cuda.synchronize()
        sps = (time.time() - t_ep) / max(n_ep, 1)
        if not stop:
            epoch += 1
        report(dict(epoch=epoch, step=step, lr=sched.get_last_lr()[0], s_per_step=sps, **validate(),
                    minutes=(time.time() - t_start) / 60))
        save(partial=stop)
    print(f"done {a.name}: epoch {epoch}, step {step}, {(time.time() - t_start) / 60:.1f} min -> {out_pt}", flush=True)


if __name__ == "__main__":
    main()
