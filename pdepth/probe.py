"""Erasure probes: remove part of the figure and let the background show through.

The background of every stimulus is the renderer's own noise texture (pdepth/render.py, texture_image), so an erased
region is filled with exactly the background that would be there without the figure: the image is what the remaining
part of the figure would look like on its own. Erasing junction j removes cubes 5j-1, 5j and 5j+1 (the end of the
incoming bar, the corner cube and the start of the outgoing bar), which breaks the loop there.
"""
import numpy as np
from scipy.ndimage import binary_dilation

from pdepth.render import periodic_noise

N_BAR = 5
DILATE = 6                                   # px: the hole also takes the cubes' anti-aliased outline


def background(st):
    """the stimulus's own background texture (set A uses texture seed `tex`; the other sets seed 0)"""
    tex = int(st["tex"]) if "tex" in st else 0
    with np.errstate(divide="ignore"):
        back = periodic_noise(st["img"].shape[0], seed=7 + 10 * tex, stretch=1.0)
    col = (0.45 * (0.8 + 0.4 * back))[..., None] * np.array([0.9, 0.95, 1.0])
    img = (np.clip(col, 0, 1) * 255).round().astype(np.uint8)
    return img[:, ::-1].copy() if bool(st["mirror"]) else img


def junction_cubes(j, n_cubes=15):
    c = N_BAR * j
    return [(c - 1) % n_cubes, c, (c + 1) % n_cubes]


def erase(st, cubes, dilate=DILATE):
    """image with the given cubes erased, and the erased pixel mask"""
    hole = binary_dilation(np.isin(st["cube"], cubes), iterations=dilate)
    img = st["img"].copy()
    img[hole] = background(st)[hole]
    return img, hole
