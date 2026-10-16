"""Random POSSIBLE cube scenes in the stimuli's style, for training read-outs and heads (never the test figure).

A scene is one or two self-avoiding open chains of straight runs of unit cubes, rendered orthographically along
v = (1,1,1)/sqrt(3) with the stimuli's shading, wood grain along each run and noise background, at a random rotation,
scale and shading permutation, half of them mirrored. Excluded by construction:
- the Penrose family (three mutually orthogonal runs of equal length), so the test figure and its controls are unseen;
- false contacts: two cubes that do not touch in 3D but whose footprints share an edge in the image (offset
  +-(e_a - e_b) + m(1,1,1), m != 0). This is how a hidden gap looks like a joint, so no training image contains one.
Labels: exact depth (pixel units, larger = farther), the exact depth gradient g = (d depth/dx, d depth/dy) with x right
and y up (dimensionless; the faces of a cube have |g| = sqrt(2)), and occlusion edges (depth jumps).
"""
import numpy as np
from scipy.ndimage import binary_dilation

from pdepth.render import periodic_noise, V, E1, E2, FACE_SHADE, WOOD

AX = np.eye(3)
G_UW = np.array([[AX[a] @ E1 / (AX[a] @ V), AX[a] @ E2 / (AX[a] @ V)] for a in range(3)])   # face a: gradient on (u, w)
DIRS = [np.array(d) for d in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
PERMS = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]


def random_chain(rng, n_runs=(2, 4), run_len=(2, 7), max_tries=50):
    """self-avoiding open chain of straight runs: cube positions [N, 3] and each cube's run axis"""
    for _ in range(max_tries):
        p = np.zeros(3, int); pos = [p.copy()]; axis = [0]; occ = {tuple(p)}; last = None; ok = True
        nr = rng.integers(n_runs[0], n_runs[1] + 1)
        lens = rng.integers(run_len[0], run_len[1] + 1, nr)
        for r in range(nr):
            choices = [d for d in DIRS if last is None or abs(d @ last) == 0]
            d = choices[rng.integers(len(choices))]
            for _ in range(lens[r]):
                p = p + d
                if tuple(p) in occ:
                    ok = False; break
                occ.add(tuple(p)); pos.append(p.copy()); axis.append(int(np.argmax(np.abs(d))))
            if not ok:
                break
            if r == 0:
                axis[0] = axis[1]
            last = d
        steps = [int(np.argmax(np.abs(pos[i + 1] - pos[i]))) for i in range(len(pos) - 1)]
        penrose = nr == 3 and len(set(steps)) == 3 and len(set(lens.tolist())) == 1
        if ok and len(pos) >= 3 and not penrose:
            return np.array(pos), np.array(axis)
    return None


def false_contact(pos):
    """two cubes that do not touch in 3D (Chebyshev distance >= 2) but whose footprints share an edge or coincide"""
    d = pos[:, None, :] - pos[None, :, :]
    far = np.abs(d).max(-1) >= 2
    dperp = d - d.mean(-1, keepdims=True)                       # drop the component along (1,1,1)
    bad = np.zeros(far.shape, bool)
    for o in [AX[a] - AX[b] for a in range(3) for b in range(3) if a != b] + [np.zeros(3)]:
        bad |= np.all(np.abs(dperp - o) < 1e-6, -1)
    return bool(np.any(bad & far))


def random_scene(rng, n_chains=(1, 2)):
    for _ in range(100):
        P, A = [], []
        for c in range(rng.integers(n_chains[0], n_chains[1] + 1)):
            ch = random_chain(rng)
            if ch is None:
                continue
            off = rng.integers(-4, 5, 3) if c else np.zeros(3, int)
            P.append(ch[0] + off); A.append(ch[1])
        if not P:
            continue
        pos, ax = np.concatenate(P), np.concatenate(A)
        if len({tuple(p) for p in pos}) < len(pos) or false_contact(pos):
            continue
        return pos.astype(float), ax
    raise RuntimeError("no valid scene")


def raycast_fast(pos, u, w):
    """pdepth.render.raycast restricted to each cube's footprint bounding box (identical output, much faster)"""
    best_s = np.full(u.shape, -np.inf); cube = np.full(u.shape, -1); face = np.full(u.shape, -1)
    J = np.array([[u[0, 1] - u[0, 0], u[1, 0] - u[0, 0]], [w[0, 1] - w[0, 0], w[1, 0] - w[0, 0]]])
    Ji = np.linalg.inv(J)
    H, W = u.shape
    offs = np.array(list(np.ndindex(2, 2, 2)), float)
    for c, lo in enumerate(pos):
        cr = Ji @ np.stack([(lo + offs) @ E1 - u[0, 0], (lo + offs) @ E2 - w[0, 0]])   # (col, row) of the vertices
        c0, c1 = int(max(np.floor(cr[0].min()) - 1, 0)), int(min(np.ceil(cr[0].max()) + 2, W))
        r0, r1 = int(max(np.floor(cr[1].min()) - 1, 0)), int(min(np.ceil(cr[1].max()) + 2, H))
        if c0 >= c1 or r0 >= r1:
            continue
        sl = (slice(r0, r1), slice(c0, c1))
        q = u[sl][..., None] * E1 + w[sl][..., None] * E2
        a = (lo - q) / V; b = (lo + 1 - q) / V
        smin, smax = a.max(-1), b.min(-1)
        hit = (smin <= smax) & (smax > best_s[sl])
        best_s[sl] = np.where(hit, smax, best_s[sl])
        cube[sl] = np.where(hit, c, cube[sl])
        face[sl] = np.where(hit, b.argmin(-1), face[sl])
    return best_s, cube, face


def render_scene(pos, ax, size=512, ss=2, angle=0.0, shift=(0.0, 0.0), perm=(0, 1, 2), tex_seed=0, bg=0.45,
                 period=5.0, cube_px=60.0, edge_px=4.0):
    """image, depth (pixel units), exact gradient g [2, H, W] (x right, y up), occlusion edges, fg, cube, face"""
    verts = np.concatenate([pos + np.array(o) for o in np.ndindex(2, 2, 2)])
    cu, cw = (verts @ E1).mean(), (verts @ E2).mean()
    half = size / 2 / cube_px                                  # cube units per half-image
    t = (np.arange(size * ss) + 0.5) / (size * ss) * 2 - 1
    X, Y = np.meshgrid(t * half, -t * half)
    X, Y = X - shift[0] * half, Y - shift[1] * half
    th = np.deg2rad(angle)
    u = cu + np.cos(th) * X + np.sin(th) * Y
    w = cw - np.sin(th) * X + np.cos(th) * Y
    px = 2 * half / (size * ss)                                # cube units per supersampled pixel
    s, cube, face = raycast_fast(pos, u, w)
    fg = cube >= 0
    depth = np.where(fg, -s, np.nan)
    p3 = u[..., None] * E1 + w[..., None] * E2 + np.nan_to_num(s, neginf=0.0)[..., None] * V
    loc = np.mod(p3, period) / period
    tiles = [periodic_noise(256, seed=b + 10 * tex_seed) for b in range(3)]
    shades = {a: FACE_SHADE[perm[a]] for a in range(3)}
    col = np.zeros(fg.shape + (3,))
    runax = np.where(fg, ax[np.where(fg, cube, 0)], 0)
    for a in range(3):
        for b in range(3):
            m = fg & (face == a) & (runax == b)
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
        back = periodic_noise(size, seed=7 + 10 * tex_seed, stretch=1.0)
    back = np.repeat(np.repeat(back, ss, 0), ss, 1)
    col[~fg] = (bg * (0.8 + 0.4 * back[~fg]))[:, None] * np.array([0.9, 0.95, 1.0])
    img = np.clip(col.reshape(size, ss, size, ss, 3).mean((1, 3)), 0, 1)
    # occlusion edges: depth jumps much larger than the step across a crease
    tau = edge_px * px * np.sqrt(2)
    dd = np.nan_to_num(depth, nan=1e6)
    jump = np.zeros(fg.shape, bool)
    jh = np.abs(dd[:, 1:] - dd[:, :-1]) > tau; jv = np.abs(dd[1:] - dd[:-1]) > tau
    jump[:, 1:] |= jh; jump[:, :-1] |= jh; jump[1:] |= jv; jump[:-1] |= jv
    sl = (slice(ss // 2, None, ss), slice(ss // 2, None, ss))
    fgd = fg[sl]
    disc = jump.reshape(size, ss, size, ss).any((1, 3)) & binary_dilation(fgd, iterations=1)
    f = np.where(fg, face, 0)[sl]
    gu, gw = G_UW[f, 0], G_UW[f, 1]
    gx = np.cos(th) * gu - np.sin(th) * gw                     # image x (right) = cos * u - sin * w
    gy = np.sin(th) * gu + np.cos(th) * gw                     # image y (up)
    grad = np.where(fgd[None], np.stack([gx, gy]), np.nan)
    return dict(img=(img * 255).round().astype(np.uint8), depth=(depth[sl] / (px * ss)).astype(np.float32),
                grad=grad.astype(np.float32), disc=disc, fg=fgd, cube=np.where(fgd, cube[sl], -1).astype(np.int16),
                face=np.where(fgd, face[sl], -1).astype(np.int8), px=px * ss)


def sample(seed, size=512):
    """one random scene with random appearance; half of them mirrored"""
    rng = np.random.default_rng(seed)
    pos, ax = random_scene(rng)
    out = render_scene(pos, ax, size=size, angle=rng.uniform(0, 360), shift=tuple(rng.uniform(-0.2, 0.2, 2)),
                       perm=PERMS[rng.integers(6)], tex_seed=int(rng.integers(1000)), bg=rng.uniform(0.3, 0.6),
                       period=rng.uniform(3, 7), cube_px=rng.uniform(40, 85) * size / 512)
    out["mirror"] = bool(rng.random() < 0.5)
    if out["mirror"]:
        for k in ("img", "depth", "fg", "cube", "face", "disc"):
            out[k] = np.flip(out[k], axis=1).copy()
        g = np.flip(out["grad"], axis=2).copy(); g[0] *= -1; out["grad"] = g
    out["n_cubes"] = len(pos)
    return out
