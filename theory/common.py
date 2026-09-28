"""shared helpers for the theory checks (reads pilot data, never writes there)"""
import os, sys, glob, json, fnmatch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))                # repository root (pdepth/)
PILOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_compat")   # bundled pilot read-out + stimulus
sys.path.insert(0, PILOT)
from tears import cube_profile, tear_readout  # noqa: E402  (pilot read-out, imported unchanged)

OUT = os.path.dirname(os.path.abspath(__file__))
PRED_BUNDLE = os.path.join(PILOT, "pred_profiles.json")


def pilot_profiles(pattern):
    """(name, 15-cube profile) of the pilot network outputs whose file name matches pattern, sorted by name.
    Uses the maps in pilot_compat/pred/ if present (not in the repository; they also need
    pilot_compat/stim/texture_a90_possible0.npz), else the profiles bundled in
    pilot_compat/pred_profiles.json (made from the same maps by pilot_compat/make_pred_profiles.py)."""
    files = sorted(glob.glob(os.path.join(PILOT, "pred", pattern)))
    if files:
        out = []
        for f in files:
            nm = os.path.basename(f)[:-4]
            st = load("texture_a90_possible0" if "possible0" in nm else "texture_a90")
            out.append((nm, cube_profile(np.load(f), st["cube"], st["n"])))
        return out
    B = json.load(open(PRED_BUNDLE))["profile"]
    return [(nm, np.array(B[nm])) for nm in sorted(B) if fnmatch.fnmatchcase(nm + ".npy", pattern)]


def object_profiles(stim):
    """the three objects' 15-cube profiles on a pilot stimulus (bundled for the possible control, not included)"""
    if os.path.exists(os.path.join(PILOT, "stim", f"{stim}.npz")):
        st = load(stim)
        return np.array([cube_profile(st["depth"][k], st["cube"], st["n"]) for k in range(3)])
    return np.array(json.load(open(PRED_BUNDLE))["objects"][stim])


def pilot_shares(pattern):
    """object share of the frame's 2/98 depth range for each pilot output matching pattern (maps or bundle)"""
    files = sorted(glob.glob(os.path.join(PILOT, "pred", pattern)))
    if files:
        fg = load("texture_a90")["fg"]
        out = []
        for f in files:
            d = np.load(f); a, b = np.percentile(d, [2, 98]); lo, hi = np.percentile(d[fg], [2, 98])
            out.append((hi - lo) / (b - a))
        return out
    B = json.load(open(PRED_BUNDLE))["share98"]
    return [B[nm] for nm in sorted(B) if fnmatch.fnmatchcase(nm + ".npy", pattern)]


def ensure(paths, cmd, log=None):
    """run an earlier check (cmd = script and arguments) if any of the files it writes is missing; its output goes
    to log (relative to this folder) or is discarded"""
    import subprocess
    missing = [p for p in paths if not os.path.exists(os.path.join(OUT, p))]
    if not missing:
        return
    print(f"[{', '.join(missing)} missing: running {' '.join(cmd)} first]", file=sys.stderr, flush=True)
    fh = open(os.path.join(OUT, log), "w") if log else open(os.devnull, "w")
    with fh:
        subprocess.run([sys.executable] + list(cmd), cwd=OUT, stdout=fh, stderr=subprocess.STDOUT, check=True)


def paper_profile(v, st):
    """the paper's loop profile (pdepth.readout: per-cube median over pixels >= 3 px inside the cube) of a
    foreground vector v"""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from pdepth.readout import cube_masks, cube_profile as med_profile
    if "_masks" not in st:
        st["_masks"] = cube_masks(st["cube"], st["n"])
    return med_profile(to_img(v, st["fg"]), st["_masks"])


def load(name="texture_a90"):
    st = dict(np.load(os.path.join(PILOT, "stim", f"{name}.npz")))
    st["n"] = int(st["n_cubes"])
    return st


def split_of(depth_img, st):
    return tear_readout(depth_img, st["cube"], st["corners"], st["n"])["split"]


def excess_of(depth_img, st):
    return tear_readout(depth_img, st["cube"], st["corners"], st["n"])["excess"]


def to_img(v, fg):
    """foreground vector -> image with nan background"""
    im = np.full(fg.shape, np.nan)
    im[fg] = v
    return im


def mad_norm(x):
    """MiDaS / Depth Anything normalisation: (x - median) / mean|x - median|"""
    m = np.median(x)
    return (x - m) / np.mean(np.abs(x - m))


def pct_norm(x, lo=2, hi=98):
    """Marigold / Lotus target normalisation to [-1, 1] with the 2/98 percentiles"""
    a, b = np.percentile(x, [lo, hi])
    return ((x - a) / (b - a) - 0.5) * 2


def ls_align(f, d):
    """least-squares scale and shift of f onto d (MiDaS ssimse)"""
    A = np.stack([f, np.ones_like(f)], 1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    return A @ c


def wmedian3(vals, p):
    """pointwise weighted median of 3 candidate maps vals [3, M] with weights p (lower median at ties)"""
    order = np.argsort(vals, 0)
    sv = np.take_along_axis(vals, order, 0)
    sp = np.asarray(p)[order]
    cum = np.cumsum(sp, 0)
    idx = (cum >= 0.5 - 1e-12).argmax(0)
    return np.take_along_axis(sv, idx[None], 0)[0]
