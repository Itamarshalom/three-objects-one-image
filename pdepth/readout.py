"""Per-cube depth profile around the loop, the input of the paper's read-out (pdepth/loop.py).

Depth is summarised per cube by its median over pixels at least ERODE px inside the cube (the visible pixels are the
same in all three objects), giving a 15-value profile. readout() also stores an object-mixture fit and a commitment
index (w, scale, r2, z, commit, corner, bar_slope) in every CSV row; these come from an earlier read-out and are not
used in the paper.
"""
import numpy as np
from scipy.ndimage import binary_erosion

ERODE = 3
ROOTS = np.exp(2j * np.pi * np.arange(3) / 3)


def cube_masks(cube, n_cubes, erode=ERODE):
    """eroded per-cube pixel masks (falls back to a lighter erosion for thin slivers)"""
    out = []
    for c in range(n_cubes):
        m = cube == c
        for e in (erode, 1, 0):
            me = binary_erosion(m, iterations=e) if e else m
            if me.sum() >= 20 or e == 0:
                break
        out.append(me)
    return out


def cube_profile(depth, masks):
    return np.array([np.nanmedian(depth[m]) if m.any() else np.nan for m in masks])


def _simplex_grid(step=0.01):
    k = int(round(1 / step))
    w = [(i, j, k - i - j) for i in range(k + 1) for j in range(k + 1 - i)]
    return np.array(w, float) / k


_W = _simplex_grid()


def mixture_fit(profile, gt_profiles):
    """profile ~ a * sum_k w_k g_k + b, w on the simplex (0.01 grid), scale a of either sign.
    returns w, a (in units of the ground-truth scale; a < 0 = depth-inverted), R^2"""
    ok = np.isfinite(profile) & np.all(np.isfinite(gt_profiles), 0)
    y = profile[ok] - profile[ok].mean()
    G = np.stack([g[ok] - g[ok].mean() for g in gt_profiles])
    M = _W @ G
    mm = (M * M).sum(1) + 1e-12
    a = (M @ y) / mm
    sse = (y * y).sum() - a * (M @ y)
    i = int(np.argmin(sse))
    return dict(w=_W[i], scale=float(a[i]), r2=float(1 - sse[i] / ((y * y).sum() + 1e-12)))


def bar_slopes(profile, n):
    """slope of each bar's profile (corner cube excluded), per cube step"""
    s = []
    for b in range(3):
        idx = np.arange(b * (n - 1) + 1, (b + 1) * (n - 1))
        y = profile[idx]
        s.append(np.polyfit(np.arange(len(idx)), y, 1)[0] if np.isfinite(y).all() else np.nan)
    return np.array(s)


def readout(depth, stim, masks=None):
    """all read-outs of one depth map (larger = farther) on one stimulus dict"""
    n_cubes = int(stim["n_cubes"]); n = n_cubes // 3 + 1
    masks = masks if masks is not None else cube_masks(stim["cube"], n_cubes)
    prof = cube_profile(depth, masks)
    gts = [cube_profile(stim["depth"][k], masks) for k in range(3)]
    mix = mixture_fit(prof, gts)
    z = (mix["w"] * ROOTS).sum()
    true_step = np.median(np.diff(gts[0][1:n - 1]))                    # one cube step of the true ramp
    slopes = bar_slopes(prof, n) / (true_step * (mix["scale"] if mix["scale"] != 0 else np.nan))
    return dict(profile=prof, w=mix["w"], scale=mix["scale"], r2=mix["r2"], z=z, commit=float(abs(z)),
                corner=int(np.argmax(mix["w"])), bar_slope=slopes)
