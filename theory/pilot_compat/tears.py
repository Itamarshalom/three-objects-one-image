"""Scale- and shift-free read-out of where a depth map tears the Penrose loop.

Depth is averaged per cube (same visible pixels in every object), giving a profile around the
loop. Object k is a steady ramp with one jump just before its corner cube. We measure, at each
corner, the excess of the step over the median step along the bars (the tear), and report the
split of the total tear over the three corners (p-hat) and the share of the loop's mismatch that
is torn at corners rather than spread along the bars.
"""
import numpy as np


def cube_profile(depth, cube, n_cubes):
    """mean depth per cube over its visible pixels"""
    return np.array([np.nanmean(depth[cube == c]) if (cube == c).any() else np.nan for c in range(n_cubes)])


def tear_readout(depth, cube, corners, n_cubes):
    prof = cube_profile(depth, cube, n_cubes)
    steps = np.roll(prof, -1) - prof                     # step from cube i to cube i+1 (cyclic)
    into_corner = (np.asarray(corners) - 1) % n_cubes    # step that enters each corner cube
    bar = np.setdiff1d(np.arange(n_cubes), into_corner)
    ramp = np.median(steps[bar])                         # typical step along a bar
    excess = steps[into_corner] - ramp                   # extra jump at each corner
    total = excess.sum()                                 # = -(n_cubes * ramp) if bars are straight ramps
    split = excess / total if abs(total) > 1e-12 else np.full(3, np.nan)
    torn_share = np.abs(excess).sum() / (np.abs(steps - 0).sum() + 1e-12)
    return dict(profile=prof, steps=steps, ramp=ramp, excess=excess, split=split, torn_share=torn_share)


def affine_fit_err(pred, gt, mask):
    """relative RMSE after the best scale and shift of pred onto gt (lower = better match)"""
    x, y = pred[mask], gt[mask]
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = A @ coef - y
    return np.sqrt((r ** 2).mean()) / (y.std() + 1e-12), coef


def match_objects(pred, stim):
    """fit error of pred against each valid object and against the equal-weight average"""
    m = stim["fg"] & np.isfinite(pred)
    cands = list(stim["depth"]) + [stim["depth"].mean(0)]
    return np.array([affine_fit_err(pred, g, m)[0] for g in cands])   # [obj1, obj2, obj3, average]
