"""Render the closed POSSIBLE rings (data/loops), the controls of the erasure probe for the Penrose loop.

Two loops of 5-cube bars that close in 3D, so every junction is visible and joined and the local cues integrate with
no obstruction: a skew hexagon H (+x, -z, +y, -x, +z, -y; 30 cubes, 6 bars) and a planar rhombus R (+x, +y, -x, -y;
20 cubes, 4 bars). Each is rendered like the stimuli (orthographic along v = (1,1,1)/sqrt(3), face shading, wood grain
along each bar, noise background) at 8 layouts (0-315 deg, 45 deg) x 2 textures (with their shading permutation and
background) x 2 chiralities (mirror) = 32 images per loop, 64 in all.
Each file stores img (uint8), depth [1, H, W] (float32, larger = farther, nan off the object), fg, the cube-id map,
n_cubes, n_bars, corners (the first cube of each bar), set (H or R), angle, tex, mirror, px (pixel size), perm = 0 and
possible = -1. Read by scripts/run_probe.py (PROBE_TAG=_loops), run_features.py and run_heads.py;
scripts/analyze_loops.py takes the exact sign of each bar's slope from depth.
usage: python scripts/make_loops.py [out_dir]   (default data/loops)
"""
import os, sys, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.render import V, E1, E2, raycast, periodic_noise, FACE_SHADE, WOOD

AXES = np.eye(3)
LOOPS = {"H": [(0, 1), (2, -1), (1, 1), (0, -1), (2, 1), (1, -1)],      # skew hexagon
         "R": [(0, 1), (1, 1), (0, -1), (1, -1)]}                        # planar rhombus
N_STEP = 5                                                               # cubes per bar


def loop(spec):
    """cube positions [N, 3] and each cube's bar axis; the loop must close"""
    p, pos, axes = np.zeros(3), [], []
    for a, s in spec:
        for _ in range(N_STEP):
            pos.append(p.copy()); axes.append(a)
            p = p + s * AXES[a]
    assert np.allclose(p, 0), "loop does not close"
    return np.array(pos), np.array(axes)


def render_cubes(pos, axes, size=512, angle_deg=0.0, ss=2, margin=0.08, tex_seed=0, shade_perm=(0, 1, 2),
                 bg=0.45, bg_seed=7, period=5):
    """the stimuli's look for any set of unit cubes (pdepth/render.py), with the grain along each cube's bar axis"""
    verts = np.concatenate([pos + np.array(o) for o in np.ndindex(2, 2, 2)])
    cu, cw = (verts @ E1).mean(), (verts @ E2).mean()
    half = np.hypot(verts @ E1 - cu, verts @ E2 - cw).max() / (1 - 2 * margin)
    t = (np.arange(size * ss) + 0.5) / (size * ss) * 2 - 1
    X, Y = np.meshgrid(t * half, -t * half)
    th = np.deg2rad(angle_deg)
    u = cu + np.cos(th) * X + np.sin(th) * Y
    w = cw - np.sin(th) * X + np.cos(th) * Y
    s, cube, face = raycast(pos, u, w)
    fg = cube >= 0
    shades = {a: FACE_SHADE[shade_perm[a]] for a in range(3)}
    p3 = u[..., None] * E1 + w[..., None] * E2 + np.where(fg, s, 0)[..., None] * V
    loc = np.mod(p3, period) / period
    tiles = [periodic_noise(256, seed=b + 10 * tex_seed) for b in range(3)]
    col = np.zeros(fg.shape + (3,))
    bar = np.where(fg, axes[np.where(fg, cube, 0)], 0)
    for a in range(3):
        for b in range(3):
            m = fg & (face == a) & (bar == b)
            if not m.any():
                continue
            o = [c for c in range(3) if c != a]
            along = b if b != a else o[0]
            across = o[0] if o[0] != along else o[1]
            i = np.clip((loc[..., across] * 256).astype(int), 0, 255)
            j = np.clip((loc[..., along] * 256).astype(int), 0, 255)
            g = tiles[b][i, j]
            col[m] = (shades[a] * (0.8 + 0.4 * g[m]))[:, None] * WOOD / WOOD.max()
    with np.errstate(divide="ignore"):
        back = periodic_noise(size, seed=bg_seed, stretch=1.0)
    back = np.repeat(np.repeat(back, ss, 0), ss, 1)
    col[~fg] = (bg * (0.8 + 0.4 * back[~fg]))[:, None] * np.array([0.9, 0.95, 1.0])
    img = np.clip(col.reshape(size, ss, size, ss, 3).mean((1, 3)), 0, 1)
    sl = (slice(ss // 2, None, ss), slice(ss // 2, None, ss))
    depth = np.where(fg, -s, np.nan)[sl]
    return dict(img=(img * 255).round().astype(np.uint8), depth=depth.astype(np.float32), fg=fg[sl],
                cube=cube[sl].astype(np.int16), face=face[sl].astype(np.int8), px=2 * half / size)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/loops"
    os.makedirs(out, exist_ok=True)
    for name, spec in LOOPS.items():
        pos, axes = loop(spec)
        for ang in range(0, 360, 45):
            for tex, perm in [(0, (0, 1, 2)), (1, (1, 2, 0))]:
                r = render_cubes(pos, axes, angle_deg=ang, tex_seed=tex, shade_perm=perm, bg_seed=7 + 10 * tex)
                for mirror in (False, True):
                    a = {k: (np.flip(r[k], axis=1).copy() if mirror else r[k]) for k in ("img", "depth", "fg", "cube")}
                    fn = f"{name}_a{ang:03d}_t{tex}{'_m' if mirror else ''}.npz"
                    np.savez_compressed(os.path.join(out, fn), depth=a["depth"][None], n_cubes=len(pos),
                                        n_bars=len(spec), corners=np.arange(0, len(pos), N_STEP), set=name, angle=ang,
                                        tex=tex, perm=0, mirror=mirror, possible=-1, px=r["px"], img=a["img"],
                                        fg=a["fg"], cube=a["cube"])
                print(name, ang, tex, flush=True)
