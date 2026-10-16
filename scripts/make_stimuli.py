"""Render the stimulus sets (v2).

A  layout sweep  : impossible image, 24 layouts (0-115 deg, 5 deg; the outline has period 120)
                   x 3 cyclic shade permutations x 2 texture seeds x 2 chiralities (mirror)      = 288
B  possible      : object k as it really looks, k = 0..2 (objects 1-3 of the report),
                   8 layouts x 2 chiralities                                                      = 48
C  gap morph     : a fraction tau of object k's trimmed stub revealed, tau in TAUS, k = 0..2,
                   4 layouts                                                                       = 96
D  nuisance      : impossible image shifted up/down by 38 px (0.15 of the half-width) or zoomed to 0.7,
                   4 layouts                                                                      = 12
E  no material   : flat-shaded and outline styles, 8 layouts                                       = 16
Each file stores the image (uint8), the three objects' depths (float32, nan off the object), the
cube-id map, the on-screen corner positions, and the factors.
usage: python scripts/make_stimuli.py [out_dir] [n_workers]
"""
import os, sys, itertools, numpy as np
from multiprocessing import Pool
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.render import render, corner_screen_xy

N_BARS = 6
PERMS = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
TAUS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/stim2"


def jobs():
    J = []
    for ang, pi, tex in itertools.product(range(0, 120, 5), range(3), (0, 1)):
        J.append(dict(set="A", angle=ang, perm=pi, tex=tex, mirror_too=True))
    for ang, k in itertools.product(range(0, 120, 15), range(3)):
        J.append(dict(set="B", angle=ang, possible=k, mirror_too=True))
    for ang, k, tau in itertools.product((0, 30, 60, 90), range(3), TAUS):
        J.append(dict(set="C", angle=ang, possible=k, stub=tau))
    for ang, (sh, zm, lab) in itertools.product((0, 30, 60, 90), [((0, 0.15), 1.0, "up"), ((0, -0.15), 1.0, "down"), ((0, 0), 0.7, "small")]):
        J.append(dict(set="D", angle=ang, shift=sh, zoom=zm, nuis=lab))
    for ang, style in itertools.product(range(0, 120, 15), ("flat", "lines")):
        J.append(dict(set="E", angle=ang, style=style))
    return J


def name_of(j, mirror=False):
    s = f"{j['set']}_a{j['angle']:03d}"
    if j["set"] == "A": s += f"_p{j['perm']}_t{j['tex']}"
    if j["set"] in "BC": s += f"_k{j['possible']}"
    if j["set"] == "C": s += f"_tau{int(round(j['stub'] * 100)):03d}"
    if j["set"] == "D": s += f"_{j['nuis']}"
    if j["set"] == "E": s += f"_{j['style']}"
    return s + ("_m" if mirror else "")


def run(j):
    name = name_of(j)
    if os.path.exists(os.path.join(OUT, name + ".npz")):
        return name
    st = render(n=N_BARS, size=512, angle_deg=j["angle"], style=j.get("style", "texture"), bg=0.45 if j.get("style", "texture") == "texture" else 0.25,
                possible=j.get("possible"), stub=j.get("stub"), tex_seed=j.get("tex", 0),
                shade_perm=PERMS[j.get("perm", 0)], shift=j.get("shift", (0, 0)), zoom=j.get("zoom", 1.0))
    xy = corner_screen_xy(N_BARS, j["angle"], shift=j.get("shift", (0, 0)), zoom=j.get("zoom", 1.0))
    meta = dict(set=j["set"], angle=j["angle"], perm=j.get("perm", 0), tex=j.get("tex", 0),
                possible=-1 if j.get("possible") is None else j["possible"], tau=-1.0 if j.get("stub") is None else j["stub"],
                nuis=j.get("nuis", ""), style=j.get("style", "texture"))
    arrs = dict(img=(st["img"] * 255).round().astype(np.uint8), depth=st["depth"].astype(np.float32),
                fg=st["fg"], cube=st["cube"].astype(np.int16))
    for mirror in ([False, True] if j.get("mirror_too") else [False]):
        a, c = arrs, xy.copy()
        if mirror:
            a = {k: np.flip(v, axis=1 if k == "img" else -1).copy() for k, v in arrs.items()}
            c[:, 0] *= -1
        nm = name_of(j, mirror)
        np.savez_compressed(os.path.join(OUT, nm + ".npz"), corners=st["corners"], n_cubes=st["n_cubes"],
                            corner_xy=c, mirror=mirror, **meta, **a)
        Image.fromarray(a["img"]).save(os.path.join(OUT, "png", nm + ".png"))
    return name


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "png"), exist_ok=True)
    J = jobs()
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as p:
        for i, nm in enumerate(p.imap_unordered(run, J), 1):
            if i % 25 == 0 or i == len(J):
                print(f"{i}/{len(J)} {nm}", flush=True)
