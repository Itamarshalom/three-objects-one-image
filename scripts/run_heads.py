"""Run a trained head (scripts/train_heads.py, outputs/heads/<name>.pt) on the stimuli and the closed possible loops.

Writes outputs/heads_out/<name>/<stim>.npz per image, the input of pdepth/oneform.py (float32, whole image: the head
does not know the figure, so mask with the stimulus):
  g [H, W, 2]: depth gradient (x right, y up, dimensionless; oneform's layout), the 1-form head's output, or for a
    0-form head the central differences of its depth (np.gradient);
  edge [H, W]: occlusion-edge probability, calibrated (sigmoid(logit - log pos_weight), pdepth/heads.py), so 0.5 is
    the even-odds threshold;  kind: '0form' or '1form';
  depth [H, W] (0-form only): pixel units of the image, larger = farther.
For a 0-form head it then runs the existing scripts under the REGISTRY key head-<name> (pdepth/models.py): the erasure
probe scripts/run_probe.py on the impossible images and controls (results/probe_head-<name>.csv) and, with
PROBE_TAG=_loops, on data/loops (results/probe_loops_head-<name>.csv), and scripts/run_models.py on all stimuli
(results/readout_head-<name>.csv, incl. set C); --no-probe skips them. Paths are relative to the repo root (run from
there), as in the other scripts.
usage: python scripts/run_heads.py <name> [npz files or quoted globs ...] [--no-probe]
       (default inputs: "data/stim2/*.npz" "data/loops/*.npz")
"""
import os, sys, glob, time, argparse, subprocess, numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from pdepth.heads import HeadRunner, ckpt_path

p = argparse.ArgumentParser(usage=__doc__.split("usage: ")[1])
p.add_argument("name")
p.add_argument("inputs", nargs="*", default=["data/stim2/*.npz", "data/loops/*.npz"])
p.add_argument("--no-probe", action="store_true", help="0-form: skip run_probe.py / run_models.py")
p.add_argument("--probe-stim", default="data/stim2/[AB]_*.npz", help="0-form: images of the erasure probe")
p.add_argument("--probe-loops", default="data/loops/*.npz", help="0-form: closed loops of the erasure probe")
p.add_argument("--readout", default="data/stim2/*.npz", help="0-form: images of run_models.py")
p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
a = p.parse_args()
if a.name.endswith(".pt") or os.sep in a.name or "/" in a.name:
    sys.exit("give the checkpoint's name (outputs/heads/<name>.pt), not a path: the REGISTRY key is head-<name>")
files = sorted(f for x in a.inputs for f in (glob.glob(x) if any(c in x for c in "*?[") else [x]))
if not files:
    sys.exit(f"no inputs found for {a.inputs}")
torch.set_num_threads(a.threads or os.cpu_count())
runner = HeadRunner(ckpt_path(a.name))
out_dir = os.path.join("outputs/heads_out", a.name)
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()
for i, f in enumerate(files):
    with np.load(f) as z:
        img = z["img"]
    m = runner.maps(img)
    extra = {}
    if runner.kind == "0form":
        gr, gc = np.gradient(m["depth"].astype(np.float64))                  # rows grow downward, y is up
        g = np.stack([gc, -gr], -1)
        extra["depth"] = m["depth"].astype(np.float32)
    else:
        g = np.moveaxis(m["g"], 0, -1)
    # float32: in float16 the integration of oneform.edge_form would stay in float16
    np.savez_compressed(os.path.join(out_dir, os.path.basename(f)[:-4] + ".npz"), kind=runner.kind,
                        g=g.astype(np.float32), edge=m["edge"].astype(np.float32), **extra)
    if (i + 1) % 50 == 0 or i + 1 == len(files):
        print(f"{a.name}: {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
print(f"done {a.name} ({runner.kind}): {len(files)} maps -> {out_dir}", flush=True)

if runner.kind == "0form" and not a.no_probe:
    del runner                                                                # each script loads the head itself
    key = "head-" + a.name
    for script, inputs, tag in (("run_probe.py", a.probe_stim, ""), ("run_probe.py", a.probe_loops, "_loops"),
                                ("run_models.py", a.readout, "")):
        cmd = [os.path.join(REPO, "scripts", script), key, inputs]
        print(" ".join(["PROBE_TAG=" + tag] * bool(tag) + ["python"] + cmd), flush=True)
        subprocess.run([sys.executable] + cmd, env={**os.environ, "PROBE_TAG": tag}, check=True)
