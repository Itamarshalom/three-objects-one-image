"""Orthographic renderer for the three straight-bar objects that share one Penrose-triangle image.

The loop is a chain of unit cubes: n-1 steps along +x, n-1 along +y, n-1 along +z. Under the
view direction v = (1,1,1)/sqrt(3) the chain's end projects onto its start, so the image closes.
Object k starts the chain at corner k; cubes before that corner are shifted by (n-1)(1,1,1),
which is a translation along v and does not move them in the image. So every cube has the same
image footprint in all three objects and only its depth differs.

The shared image is drawn with the occlusion of a *real* joint at every corner (the familiar
drawing). For object k the false joint is at corner k; there its near bar is trimmed by a prism
parallel to v, whose cut faces are edge-on and invisible. So object k's depth map is simply the
depth of the face shown in the shared image, placed where object k puts that cube.

Gap morph: with possible=k and stub=tau in [0,1], a fraction tau of the trimmed stub of object k's
near bar is shown again (the trim plane moves along the bar). Every tau > 0 is a physical render
of object k alone; tau = 1 is object k's plain render, tau = 0 is the shared impossible image.
"""
import numpy as np

V = np.ones(3) / np.sqrt(3)                        # towards the camera
E1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)       # image right
E2 = np.cross(V, E1)                               # image up
FACE_SHADE = {0: 0.55, 1: 0.80, 2: 0.95}           # +x, +y, +z faces (shading by normal only)
WOOD = np.array([0.72, 0.52, 0.33])


def loop_cubes(n):
    """cube corners along the loop and the loop index of the three corners"""
    steps = [np.array(s, float) for s in ([1, 0, 0],) * (n - 1) + ([0, 1, 0],) * (n - 1) + ([0, 0, 1],) * (n - 1)]
    p, pos = np.zeros(3), []
    for s in steps:
        pos.append(p.copy())
        p = p + s
    return np.array(pos), [0, n - 1, 2 * (n - 1)]


def object_cubes(n, k):
    """cube positions of object k (gap at corner k): cubes before corner k move by (n-1)(1,1,1)"""
    pos, corners = loop_cubes(n)
    pos = pos.copy()
    pos[: corners[k]] += (n - 1)
    return pos


def raycast(pos, u, w):
    """nearest hit per pixel: returns (s, cube id, face axis); s = height along v, larger = nearer"""
    q = u[..., None] * E1 + w[..., None] * E2                       # [H, W, 3] points on the image plane
    best_s = np.full(u.shape, -np.inf)
    cube = np.full(u.shape, -1)
    face = np.full(u.shape, -1)
    for c, lo in enumerate(pos):
        a = (lo - q) / V                                              # s where each slab starts
        b = (lo + 1 - q) / V                                          # s where each slab ends
        smin, smax = a.max(-1), b.min(-1)
        hit = (smin <= smax) & (smax > best_s)
        best_s = np.where(hit, smax, best_s)
        cube = np.where(hit, c, cube)
        face = np.where(hit, b.argmin(-1), face)
    return best_s, cube, face


def render_side(n, k, view=(0.35, 1.0, 0.55), size=400, margin=0.06, up=(0, 0, 1)):
    """object k seen orthographically from another direction (for figures: shows where its gap is).
    returns an RGB image with flat shading by face normal on white."""
    pos = object_cubes(n, k)
    v = np.asarray(view, float); v /= np.linalg.norm(v)
    e1 = np.cross(up, v); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    verts = np.concatenate([pos + np.array(o) for o in np.ndindex(2, 2, 2)])
    uu, ww = verts @ e1, verts @ e2
    c = np.array([(uu.max() + uu.min()) / 2, (ww.max() + ww.min()) / 2])
    half = max(uu.max() - uu.min(), ww.max() - ww.min()) / 2 / (1 - 2 * margin)
    t = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(c[0] + t * half, c[1] - t * half)
    q = X[..., None] * e1 + Y[..., None] * e2
    best = np.full(X.shape, -np.inf); face = np.full(X.shape, -1)
    sgn = np.sign(v)
    for lo in pos:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (lo - q) / v; b = (lo + 1 - q) / v
        smin, smax = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
        hit = (smin <= smax) & (smax > best)
        best = np.where(hit, smax, best)
        face = np.where(hit, np.maximum(a, b).argmin(-1), face)
    shade = np.array([0.55, 0.80, 0.95])
    img = np.where(face >= 0, shade[np.clip(face, 0, 2)], 1.0)
    rgb = img[..., None] * np.where(face[..., None] >= 0, WOOD / WOOD.max(), 1.0)
    return np.clip(rgb, 0, 1)


def trim_lookup(n, k, res=1024, margin=0.12):
    """raster, in image-plane coordinates (u, w), of object k's trim: where its own render would show its near
    bar in front of the shared image's surface. Returns (u0, w0, step, removed mask, s_shared) so that a 3D
    point p is trimmed away iff removed[proj p] and p.V > s_shared[proj p] (it lies in front of that surface)."""
    u, w = pixel_grid(n, res, 0.0, margin)
    objs = [object_cubes(n, j) for j in range(3)]
    casts = [raycast(p, u, w) for p in objs]
    pos0, corners = loop_cubes(n)
    cc = [pos0[ci] + 0.5 for ci in corners]
    d = np.stack([np.hypot(u - c @ E1, w - c @ E2) for c in cc])
    use = (d.argmin(0) + 1) % 3
    cube = np.choose(use, [c[1] for c in casts]); face = np.choose(use, [c[2] for c in casts])
    fg = cube >= 0
    q = u[..., None] * E1 + w[..., None] * E2
    lo = objs[k][np.where(fg, cube, 0)]
    s_shared = np.take_along_axis((lo + 1 - q) / V, np.where(fg, face, 0)[..., None], -1)[..., 0]
    s_own = casts[k][0]
    removed = fg & (casts[k][1] >= 0) & (s_own > s_shared + 1e-6)
    return u[0, 0], w[0, 0], (u[0, 1] - u[0, 0], w[1, 0] - w[0, 0]), removed, s_shared


def render_side_trimmed(n, k, view, up=(0, 0, 1), size=300, margin=0.06, steps=700):
    """object k WITH its trim, seen orthographically along `view`, by ray marching (for Figure 1, figures/fig_hero.py).
    At the special view it reproduces the shared impossible image; off it, the trimmed bar end shows."""
    pos = object_cubes(n, k)
    u0, w0, (du, dw), removed, s_sh = trim_lookup(n, k)
    v = np.asarray(view, float); v /= np.linalg.norm(v)
    e1 = np.cross(up, v); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    verts = np.concatenate([pos + np.array(o) for o in np.ndindex(2, 2, 2)])
    c = np.array([(verts @ e1).max() + (verts @ e1).min(), (verts @ e2).max() + (verts @ e2).min()]) / 2
    half = max(np.ptp(verts @ e1), np.ptp(verts @ e2)) / 2 / (1 - 2 * margin)
    t = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(c[0] + t * half, c[1] - t * half)
    base = X[..., None] * e1 + Y[..., None] * e2
    s_hi, s_lo = (verts @ v).max() + 0.5, (verts @ v).min() - 0.5
    hit = np.zeros(X.shape, bool); hcube = np.full(X.shape, -1); cutsurf = np.zeros(X.shape, bool)
    prev_removed = np.zeros(X.shape, bool)
    H, W = removed.shape
    for s in np.linspace(s_hi, s_lo, steps):                       # march from the camera side
        p = base + s * v
        todo = ~hit
        if not todo.any():
            break
        cid = np.full(X.shape, -1)
        for ci, lo in enumerate(pos):
            rel = p - lo
            ins = np.all((rel >= 0) & (rel <= 1), -1)
            cid = np.where(ins & (cid < 0), ci, cid)
        inside = cid >= 0
        # trimmed away: projection falls in the removed region and the point is in front of the shared surface
        iu = np.clip(((p @ E1 - u0) / du).round().astype(int), 0, W - 1)
        iw = np.clip(((p @ E2 - w0) / dw).round().astype(int), 0, H - 1)
        cut = removed[iw, iu] & ((p @ V) > s_sh[iw, iu] + 0.03)   # tolerance keeps the shared surface solid
        new = todo & inside & ~cut
        hcube = np.where(new, cid, hcube)
        cutsurf = np.where(new, prev_removed, cutsurf)
        hit |= new
        prev_removed = np.where(todo, inside & cut, prev_removed)
    # entry face of the hit cube, analytically (the ray travels along -v): the slab entered last
    lo = pos[np.clip(hcube, 0, None)]
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (lo - base) / v; tb = (lo + 1 - base) / v                 # s where each slab's planes are crossed
    s_enter = np.minimum(np.maximum(ta, tb), np.inf)                   # entering from high s: the larger crossing
    axis = np.argmin(np.where(np.isfinite(s_enter), s_enter, np.inf), -1)
    face = np.where(hit, np.where(cutsurf, 6, axis + 3 * (v[axis] > 0)), -1)
    # faces 0-2 are -x,-y,-z (distance to low side), 3-5 are +x,+y,+z
    normals = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    shade = np.ones(X.shape)
    for f in range(6):
        axis = f % 3
        shade = np.where(face == f, FACE_SHADE[axis] if normals[f] @ v > 0 else 0.45, shade)
    shade = np.where(face == 6, 0.35, shade)
    rgb = shade[..., None] * np.where(hit[..., None], WOOD / WOOD.max(), 1.0)
    return np.clip(np.where(hit[..., None], rgb, 1.0), 0, 1)


def figure_centre(n):
    """image-plane centroid of the three corner cubes: the figure's 3-fold symmetry centre"""
    pos, corners = loop_cubes(n)
    c = np.array([pos[i] + 0.5 for i in corners])
    return (c @ E1).mean(), (c @ E2).mean()


def _half(n, margin):
    cu, cw = figure_centre(n)
    pos, _ = loop_cubes(n)
    verts = np.concatenate([pos + np.array(o) for o in np.ndindex(2, 2, 2)])
    return np.hypot(verts @ E1 - cu, verts @ E2 - cw).max() / (1 - 2 * margin)   # covers every rotation


def pixel_grid(n, size, angle_deg=0.0, margin=0.12, shift=(0.0, 0.0), zoom=1.0):
    """image-plane coordinates of pixel centres. Rotation is about the 3-fold centre, so a 120 deg
    rotation maps the outline onto itself. shift (x right, y up) is in units of the image half-width."""
    cu, cw = figure_centre(n)
    half = _half(n, margin)
    t = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(t * half, -t * half)                            # row 0 is the top
    X, Y = (X - shift[0] * half) / zoom, (Y - shift[1] * half) / zoom
    th = np.deg2rad(angle_deg)
    u = cu + np.cos(th) * X + np.sin(th) * Y
    w = cw - np.sin(th) * X + np.cos(th) * Y
    return u, w


def corner_screen_xy(n, angle_deg, margin=0.12, shift=(0.0, 0.0), zoom=1.0):
    """on-screen position of each corner cube centre, in units of the image half-width (x right, y up)"""
    cu, cw = figure_centre(n)
    half = _half(n, margin)
    pos, corners = loop_cubes(n)
    th = np.deg2rad(angle_deg)
    out = []
    for i in corners:
        du, dw = (pos[i] + 0.5) @ E1 - cu, (pos[i] + 0.5) @ E2 - cw
        # inverse of the map in pixel_grid
        X, Y = np.cos(th) * du - np.sin(th) * dw, np.sin(th) * du + np.cos(th) * dw
        out.append(((X * zoom) / half + shift[0], (Y * zoom) / half + shift[1]))
    return np.array(out)


def edges_of(cube, face, pos, ss, edge_px=1.5):
    """outline mask of one object's own render: face changes and jumps between face planes"""
    fg = cube >= 0
    lo = pos[np.where(fg, cube, 0)]
    plane = np.take_along_axis(lo, np.where(fg, face, 0)[..., None], -1)[..., 0]
    k = np.where(fg, face * 1000 + plane.astype(int), -1)
    edge = np.zeros_like(fg)
    edge[:-1] |= k[:-1] != k[1:]
    edge[:, :-1] |= k[:, :-1] != k[:, 1:]
    from scipy.ndimage import binary_dilation
    return binary_dilation(edge, iterations=max(int(round(edge_px * ss / 2)), 1))


def stylise(face, fg, edge, ss, style="flat", bg=0.25, shades=None):
    """flat: shading by face normal on a grey background; lines: the same plus black outlines"""
    shades = shades or FACE_SHADE
    lut = np.array([shades[0], shades[1], shades[2]])
    shade = np.where(fg, lut[np.where(fg, face, 0)], bg)
    if style == "lines":
        shade = np.where(edge, 0.0, shade)
    H = fg.shape[0] // ss
    img = shade.reshape(H, ss, H, ss).mean((1, 3))
    return np.repeat(img[..., None], 3, -1)


def periodic_noise(res=256, seed=0, slope=1.6, stretch=6.0):
    """seamless tile of random-phase 1/f noise; stretch > 1 smooths it along axis 1 (wood grain)"""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(res) * res
    FX, FY = np.meshgrid(f, f)
    r = np.sqrt((FX * stretch) ** 2 + FY ** 2)
    amp = np.where(r > 0, r ** -slope, 0)
    tile = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((res, res)))))
    return (tile - tile.min()) / (tile.max() - tile.min())


def texture_image(loc, face, cube, fg, n, ss, bg, res=256, tex_seed=0, shades=None, bg_seed=None):
    """wood bars: the grain runs along each bar (cube i belongs to the bar along axis i // (n-1)),
    so the texture is covariant with the 3-fold symmetry. Shaded by face normal, no shadows."""
    shades = shades or FACE_SHADE
    tiles = [periodic_noise(res, seed=b + 10 * tex_seed) for b in range(3)]   # one tile per bar axis
    col = np.zeros(fg.shape + (3,))
    bar = np.where(fg, cube, 0) // (n - 1)
    for a in range(3):
        for b in range(3):
            m = fg & (face == a) & (bar == b)
            if not m.any():
                continue
            o = [c for c in range(3) if c != a]
            along = b if b != a else o[0]                                # end-cap faces: any in-face axis
            across = o[0] if o[0] != along else o[1]
            i = np.clip((loc[..., across] * res).astype(int), 0, res - 1)
            j = np.clip((loc[..., along] * res).astype(int), 0, res - 1)
            g = tiles[b][i, j]
            col[m] = (shades[a] * (0.8 + 0.4 * g[m]))[:, None] * WOOD / WOOD.max()
    H = fg.shape[0] // ss
    back = periodic_noise(H, seed=(7 + 10 * tex_seed) if bg_seed is None else bg_seed, stretch=1.0)
    back = np.repeat(np.repeat(back, ss, 0), ss, 1)
    col[~fg] = (bg * (0.8 + 0.4 * back[~fg]))[:, None] * np.array([0.9, 0.95, 1.0])
    return np.clip(col.reshape(H, ss, H, ss, 3).mean((1, 3)), 0, 1)


def render(n=6, size=512, angle_deg=0.0, ss=2, style="texture", bg=0.45, possible=None, stub=None,
           tex_seed=0, shade_perm=(0, 1, 2), margin=0.12, shift=(0.0, 0.0), zoom=1.0, bg_seed=None):
    """returns dict: img [size,size,3] in 0..1, depth [3,size,size] (nan off the object, larger = farther),
    fg, own (each object's own untrimmed z-buffer depth), cube / face label maps, corners, n_cubes.
    possible=k alone: object k as it really looks (gap visible). possible=k, stub=tau: gap morph."""
    u, w = pixel_grid(n, size * ss, angle_deg, margin, shift, zoom)
    objs = [object_cubes(n, k) for k in range(3)]
    casts = [raycast(p, u, w) for p in objs]
    pos0, corners = loop_cubes(n)
    # shared labels: near corner k use an object whose joint at k is real
    cc = [pos0[ci] + 0.5 for ci in corners]
    d = np.stack([np.hypot(u - c @ E1, w - c @ E2) for c in cc])       # distance to each corner
    use = (d.argmin(0) + 1) % 3                                        # object with a real joint there
    if possible is not None:
        k = possible
        if stub is None or stub >= 1:
            use = np.full_like(use, k)
        elif stub > 0:
            # stub region: where object k's own render differs from the shared image
            differs = (casts[k][1] != np.choose(use, [c[1] for c in casts])) | \
                      (casts[k][2] != np.choose(use, [c[2] for c in casts]))
            near = (corners[k] - 1) % len(pos0)                        # object k's last (nearest) cube
            ax = near // (n - 1)                                       # axis of its bar
            e = np.eye(3)[ax]
            t = u * (e @ E1) + w * (e @ E2)                            # screen coordinate along that bar
            stub_px = differs & (casts[k][1] >= 0)
            t0, t1 = t[stub_px].min(), t[stub_px].max()
            reveal = stub_px & (t <= t0 + stub * (t1 - t0))            # the trim plane moves along the bar
            use = np.where(reveal, k, use)
    shades = {a: FACE_SHADE[shade_perm[a]] for a in range(3)}
    cube = np.choose(use, [c[1] for c in casts])
    face = np.choose(use, [c[2] for c in casts])
    fg = cube >= 0
    if style == "texture":
        # texture fixed in space with period n-1: objects differ by shifts of (n-1)(1,1,1), so all agree
        s_use = np.choose(use, [c[0] for c in casts])
        p3 = u[..., None] * E1 + w[..., None] * E2 + s_use[..., None] * V
        loc = np.mod(np.nan_to_num(p3, neginf=0.0), n - 1) / (n - 1)
        img = texture_image(loc, face, cube, fg, n, ss, bg, tex_seed=tex_seed, shades=shades, bg_seed=bg_seed)
    else:
        # outlines per object, switched like the labels, so no seam appears where objects switch
        edge = np.choose(use, [edges_of(c[1], c[2], p, ss) for c, p in zip(casts, objs)]) if style == "lines" else None
        img = stylise(face, fg, edge, ss, style, bg, shades)
    # depth of the shown face in each object: the face plane is fixed per cube, s from its slab
    q = u[..., None] * E1 + w[..., None] * E2
    depths = []
    for p in objs:
        lo = p[np.where(fg, cube, 0)]
        b = (lo + 1 - q) / V
        s = np.take_along_axis(b, np.where(fg, face, 0)[..., None], -1)[..., 0]
        depths.append(np.where(fg, -s, np.nan))                          # distance, larger = farther
    sl = (slice(ss // 2, None, ss), slice(ss // 2, None, ss))
    own = [np.where(c[1] >= 0, -c[0], np.nan)[sl] for c in casts]
    return dict(img=img, depth=np.stack([dd[sl] for dd in depths]), fg=fg[sl], own=np.stack(own),
                cube=cube[sl], face=face[sl], corners=np.array(corners), n_cubes=len(pos0))
