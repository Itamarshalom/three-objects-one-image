"""1-form read-out: depth from a field of local depth changes, with the loop's topology made explicit.

A 1-form here is a per-pixel depth gradient g [H, W, 2] = (d depth/dx, d depth/dy), x right, y up, dimensionless
(depth larger = farther), on a figure mask: exact (exact_g) or read out of a network by a probe or head. On the
4-neighbour graph of the mask it gives one depth change per edge (edge_form: trapezoid rule times the grid step, so
depth comes out in the step's units; cube units with step = pixel size).
- integrate(): least-squares depth on the graph (sparse normal equations, one free shift per connected piece), with
  edges removed (a cut) and per-edge weights. On a closed loop that carries circulation this is the L2 (Hodge)
  projection onto depth maps: the circulation is spread around the loop and the bars of the Penrose loop go flat.
- loop_closed(): is there a path once around the loop that avoids the removed pixels (occlusion edges) and cut edges?
  Cut a bar across its middle and ask whether the two ends of a cut edge still connect.
- circulation Gamma = minus the depth change once around the loop in the direction of travel (cubes 0, 1, 2, ...;
  bar j runs from corner j to corner j+1), i.e. the jump that closes it: T = 5 sqrt(3) cube units for the impossible
  image (every object comes nearer by T along its bars and jumps back by T at its gap), 0 for a closed possible loop.
  circulation_bars(): along-bar slope of each bar (bar_slopes) times the bar's image length, summed;
  circulation_path(): the edge values summed along a closed path through every cube in turn (NaN if the loop is open);
  readout()'s gamma_cut: the mean jump at the cut junction over the cuts (least squares on all edges, the most stable
  of the three when g is noisy; NaN if the loop is open).
- cut_objects(): integrate with a cut across junction k (junction_cut: on exact pixel data, the edges between cube
  5k-1 and the corner cube 5k): for the Penrose loop this gives object k, with the whole circulation as one jump at k.
readout() runs all of it on a pixel grid; patch_readout() on the 37x37 patch grid of scripts/run_features.py, where a
probe gives one gradient per patch and the majority-cube map labels the nodes.
"""
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt, median as label_median
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import spsolve

from pdepth.loop import N_BAR, bars_and_joints
from pdepth.readout import cube_masks, cube_profile

T = 5 * np.sqrt(3)                               # obstruction of the Penrose loop, cube units
SLOPE = -1 / np.sqrt(2)                          # along-bar slope of every bar of every object
G37 = 37                                         # patch grid of scripts/run_features.py
PATCH = 14 * 512 / 518                           # patch side in 512-px stimulus pixels
IDX = np.floor((np.arange(518) + 0.5) * 512 / 518).astype(int)   # 518-px input pixel -> 512-px stimulus pixel


# ---------------------------------------------------------------- graph, edge values, least squares

def graph(mask):
    """4-neighbour graph of a mask: node index map (-1 off the mask), node rows / cols (row-major order) and edges
    p -> q, q being the lower (ax 0) or right (ax 1) neighbour of p"""
    H, W = mask.shape
    idx = np.full(mask.shape, -1); idx[mask] = np.arange(mask.sum())
    p, q, ax = [], [], []
    for a, (di, dj) in enumerate(((1, 0), (0, 1))):
        m = mask[:H - di, :W - dj] & mask[di:, dj:]
        p.append(idx[:H - di, :W - dj][m]); q.append(idx[di:, dj:][m]); ax.append(np.full(m.sum(), a))
    r, c = np.nonzero(mask)
    return dict(idx=idx, r=r, c=c, n=len(r), p=np.concatenate(p), q=np.concatenate(q), ax=np.concatenate(ax),
                shape=mask.shape)


def edge_form(G, g, step):
    """depth change along every edge p -> q from g [H, W, 2] (x right, y up): trapezoid rule, step = grid spacing in
    depth units. An edge with one unknown (NaN) end uses the other end; with both unknown it is NaN (left out)."""
    gn = g[G["r"], G["c"]]
    k = np.where(G["ax"] == 1, 0, 1)                                 # right step: d/dx; down step: -d/dy
    a, b = gn[G["p"], k], gn[G["q"], k]
    v = np.where(np.isnan(a), b, np.where(np.isnan(b), a, 0.5 * (a + b)))
    return np.where(G["ax"] == 1, 1.0, -1.0) * step * v


def edge_weights(G, wmap, how=np.minimum):
    """per-edge weights from a per-pixel weight map (e.g. 1 - edge probability): the smaller of the two ends"""
    wn = wmap[G["r"], G["c"]]
    return how(wn[G["p"]], wn[G["q"]])


def _adj(G, keep):
    n = G["n"]
    return sp.csr_matrix((np.ones(int(keep.sum())), (G["p"][keep], G["q"][keep])), shape=(n, n))


def integrate(G, om, w=None, cut=None):
    """least-squares depth phi on the graph: minimise sum_e w_e (phi_q - phi_p - om_e)^2 over the kept edges (finite
    om, not in cut, w > 0). Each connected piece is fixed by its own median (= 0); single nodes are NaN.
    returns phi per node and the piece label of every node"""
    w = np.ones(len(om)) if w is None else np.asarray(w, float)
    keep = np.isfinite(om) & (w > 0) & (True if cut is None else ~cut)
    p, q, o, ww = G["p"][keep], G["q"][keep], om[keep], w[keep]
    n, m = G["n"], len(p)
    A = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[q, p])), shape=(m, n))
    L = (A.T @ sp.diags(ww) @ A).tocsr()
    b = A.T @ (ww * o)
    npc, piece = connected_components(_adj(G, keep), directed=False)
    first = np.unique(piece, return_index=True)[1]
    # pin one node per piece: b sums to zero on every piece, so this fixes the shifts and leaves the solution exact
    phi = spsolve((L + sp.csr_matrix((np.ones(npc), (first, first)), shape=(n, n))).tocsc(), b)
    size = np.bincount(piece, minlength=npc)
    big = np.nonzero(size >= 2)[0]
    shift = np.full(npc, np.nan)
    if len(big):
        shift[big] = label_median(phi, piece, big)
    return phi - shift[piece], piece


def to_map(G, v):
    out = np.full(G["shape"], np.nan)
    out[G["r"], G["c"]] = v
    return out


# ---------------------------------------------------------------- cuts and loop closure

def cut_between(G, lab, a, b):
    """edges joining a node labelled in a to a node labelled in b (lab = cube id per node)"""
    lp, lq = lab[G["p"]], lab[G["q"]]
    return (np.isin(lp, a) & np.isin(lq, b)) | (np.isin(lp, b) & np.isin(lq, a))


def junction_cut(G, lab, k, n_cubes):
    """the cut across junction k: edges from the incoming bar (cubes 5k-4..5k-1) to the corner cube 5k and the
    outgoing bar (5k..5k+4). The two sides touch only at junction k. On exact pixel data this is the edge set between
    cube 5k-1 and cube 5k (object k's gap in the Penrose loop): the other pairs that touch there (5k-1 and 5k+1) are
    occlusion edges, already removed. On the patch grid or with predicted edges the wider set keeps the cut tight."""
    c = N_BAR * k
    return cut_between(G, lab, [(c - i) % n_cubes for i in range(1, N_BAR)], [(c + i) % n_cubes for i in range(N_BAR)])


def loop_closed(G, lab, n_bars, keep=None):
    """is there a path once around the loop over the kept edges? For each bar j the kept edges across its middle (cubes
    5j+1, 5j+2 to 5j+3, 5j+4) are removed: the loop closes through bar j if the two ends of one of them still connect
    (the long way round), since that edge and the path make a cycle that crosses the bar once. A test on the graph
    itself: asking instead whether two cubes connect the long way and also the short way is fooled when one cube is in
    two pieces (an edge line across cube 5j+1 or 5j+3 cut the loop but read closed on 12 of 16 such A tests).
    returns closed (through some bar) and the per-bar flags"""
    keep = np.ones(len(G["p"]), bool) if keep is None else keep
    flags = []
    for j in range(n_bars):
        c = N_BAR * j
        s = keep & cut_between(G, lab, [c + 1, c + 2], [c + 3, c + 4])
        piece = connected_components(_adj(G, keep & ~s), directed=False)[1]
        flags.append(bool((piece[G["p"][s]] == piece[G["q"][s]]).any()))
    return bool(any(flags)), np.array(flags)


# ---------------------------------------------------------------- circulation

def bar_geometry(lab_map, n_bars, step):
    """direction of travel t_j (unit, x right, y up) and image length L_j (depth units) of every bar, from the
    centroids of its two middle cubes 5j+2, 5j+3 (L_j = N_BAR cube steps). Their visible parts are translated copies;
    the end cubes of a bar can be cut by the other bar at a corner, which biases a fit through them by ~3 deg. Use a
    pixel cube map: majority-cube centroids on the 37x37 grid are off by up to ~8 deg."""
    rr, cc = np.indices(lab_map.shape)
    t, L = np.full((n_bars, 2), np.nan), np.full(n_bars, np.nan)
    for j in range(n_bars):
        s, cen = [], []
        for i in (2, 3) if all((lab_map == N_BAR * j + i).any() for i in (2, 3)) else range(1, N_BAR):
            m = lab_map == N_BAR * j + i
            if m.any():
                s.append(i); cen.append((cc[m].mean(), -rr[m].mean()))
        if len(s) >= 2:
            d = np.polyfit(s, np.array(cen), 1)[0]                  # image displacement per cube step (x, y)
            t[j], L[j] = d / np.linalg.norm(d), N_BAR * np.linalg.norm(d) * step
    return t, L


def corner_geometry(corner_xy, half):
    """the same from the stimuli's corner_xy (corner cube centres, units of the image half-width; half = half-width in
    depth units, e.g. 256 * PX)"""
    c = np.asarray(corner_xy, float)
    d = np.roll(c, -1, 0) - c
    n = np.linalg.norm(d, axis=1)
    return d / n[:, None], n * half


def bar_slopes(g, lab_map, t, n_bars, ok=None):
    """along-bar slope of each bar, g . t_j over its interior cubes: depth change per unit image length in the
    direction of travel (every bar of every Penrose object: -1/sqrt(2)). Both long faces of a bar contain its axis, so
    they share it, but their cross-bar components are opposite (+-1.22): a plain median follows the larger face and a
    0.3 deg error in t moves it by 1%. So each face side (sign of g . t_perp) gets its own median and the two are
    averaged, which cancels direction errors to first order and weights the faces equally."""
    s = np.full(n_bars, np.nan)
    fin = np.isfinite(g).all(-1) & (True if ok is None else ok)
    for j in range(n_bars):
        m = np.isin(lab_map, N_BAR * j + np.arange(1, N_BAR)) & fin
        if not m.any():
            continue
        a, side = g[m] @ t[j], g[m] @ np.array([-t[j][1], t[j][0]]) >= 0
        s[j] = 0.5 * (np.median(a[side]) + np.median(a[~side])) if side.any() and (~side).any() else np.median(a)
    return s


def circulation_bars(s, L):
    """Gamma from the bars: minus the sum of along-bar slope x bar image length (a real joint adds no change)"""
    return float(-np.sum(s * L))


def loop_path(G, lab, n_cubes, keep=None):
    """closed node path once around the loop through cubes 0, 1, ..., n_cubes - 1, 0: shortest paths between central
    nodes of consecutive cubes (the node nearest the centre of the cube's largest connected part), each kept inside
    those two cubes (and any skipped cube between them, if a cube has no node), so it cannot go the long way round.
    None if some junction cannot be crossed (the loop is open there)."""
    keep = np.ones(len(G["p"]), bool) if keep is None else keep
    xy = np.c_[G["r"], G["c"]].astype(float)
    lp, lq = lab[G["p"]], lab[G["q"]]
    piece = connected_components(_adj(G, keep & (lp == lq)), directed=False)[1]
    way = {}
    for c in range(n_cubes):
        i = np.nonzero(lab == c)[0]
        if len(i):
            u, n = np.unique(piece[i], return_counts=True)
            i = i[piece[i] == u[np.argmax(n)]]
            way[c] = i[np.argmin(((xy[i] - xy[i].mean(0)) ** 2).sum(1))]
    cs = sorted(way)
    if len(cs) < 3:
        return None
    path = [way[cs[0]]]
    for a, b in zip(cs, cs[1:] + cs[:1]):
        win = [(a + i) % n_cubes for i in range((b - a) % n_cubes + 1)]
        A = _adj(G, keep & np.isin(lp, win) & np.isin(lq, win))
        d, pred = shortest_path(A, directed=False, unweighted=True, indices=way[a], return_predecessors=True)
        if not np.isfinite(d[way[b]]):
            return None
        seg, v = [], way[b]
        while v != way[a]:
            seg.append(v); v = pred[v]
        path += seg[::-1]
    return np.array(path)


def path_sum(G, om, path):
    """sum of the edge values along a node path (an edge walked q -> p counts -om)"""
    n = G["n"]
    O = sp.csr_matrix((np.r_[om, -om], (np.r_[G["p"], G["q"]], np.r_[G["q"], G["p"]])), shape=(n, n))
    return float(np.asarray(O[path[:-1], path[1:]]).sum())


def circulation_path(G, om, lab, n_cubes, keep=None):
    """Gamma from the edges: minus the sum of edge values once around the loop (NaN if the loop is open)"""
    keep = np.isfinite(om) & (True if keep is None else keep)
    path = loop_path(G, lab, n_cubes, keep)
    return np.nan if path is None else -path_sum(G, np.nan_to_num(om), path)


# ---------------------------------------------------------------- object recovery and the full read-out

def cut_objects(g, mask, lab_map, step, n_bars=3, ks=None, w=None, masks=None):
    """integrate g on the mask with a cut across junction k, for each k in ks (default all): list of
    (depth map [H, W], loop profile [N_BAR * n_bars]); the profile is the per-cube median over masks (default
    pdepth.readout.cube_masks of lab_map)"""
    n_cubes = N_BAR * n_bars
    G = graph(mask)
    lab = lab_map[G["r"], G["c"]]
    om = edge_form(G, g, step)
    masks = cube_masks(lab_map, n_cubes) if masks is None else masks
    out = []
    for k in (range(n_bars) if ks is None else np.atleast_1d(ks)):
        phi = to_map(G, integrate(G, om, w, junction_cut(G, lab, k, n_cubes))[0])
        out.append((phi, cube_profile(phi, masks)))
    return out


def readout(g, mask, lab_map, step, n_bars=3, w=None, masks=None, geom=None, cuts=True):
    """the whole 1-form read-out of g on the mask (figure minus occlusion-edge pixels): loop closure, per-bar slopes,
    circulation (gamma_bars, gamma_path and, with cuts, gamma_cut; depth units), the least-squares depth with no cut
    and with a cut at every junction, each with its loop profile and pdepth.loop bars/joints (J_k of cut k is the jump
    that closes the loop there; gamma_cut is their mean, NaN if the loop is open, like gamma_path; each cut also says
    whether the loop is still closed after it and whether its two sides are joined the long way round).
    geom = (t, L) overrides bar_geometry(lab_map)."""
    n_cubes = N_BAR * n_bars
    G = graph(mask)
    lab = lab_map[G["r"], G["c"]]
    om = edge_form(G, g, step)
    keep = np.isfinite(om) & (True if w is None else np.asarray(w) > 0)
    closed, flags = loop_closed(G, lab, n_bars, keep)
    t, L = bar_geometry(lab_map, n_bars, step) if geom is None else geom
    s = bar_slopes(g, np.where(mask, lab_map, -1), t, n_bars)
    masks = cube_masks(lab_map, n_cubes) if masks is None else masks
    out = dict(closed=closed, closed_bars=flags, slopes=s, t=t, L=L, gamma_bars=circulation_bars(s, L),
               gamma_path=circulation_path(G, om, lab, n_cubes, keep), n_nodes=G["n"])
    phi = to_map(G, integrate(G, om, w)[0])
    prof = cube_profile(phi, masks)
    out.update(depth=phi, profile=prof, bars_joints=bars_and_joints(prof, n_bars), cuts=[])
    for k in (range(n_bars) if cuts else []):
        cut = junction_cut(G, lab, k, n_cubes)
        phi, piece = integrate(G, om, w, cut)
        prof = cube_profile(to_map(G, phi), masks)
        # joined: the two sides of junction k are still one piece (the long way round), so its jump is measured on one
        # integrated piece; else (a cut at another junction of an open loop) the pieces are centred separately
        joined = bool(np.intersect1d(piece[lab == (N_BAR * k - 1) % n_cubes], piece[lab == N_BAR * k]).size)
        out["cuts"].append(dict(k=k, depth=to_map(G, phi), profile=prof, bars_joints=bars_and_joints(prof, n_bars),
                                closed=loop_closed(G, lab, n_bars, keep & ~cut)[0], joined=joined))
    # circulation is defined on a closed loop only (as gamma_path): on an open one (a possible control) a cut at a
    # junction other than its opening splits it and the 'jump' there is ~T/2 from the separate centring (B: 0.67 T)
    out["gamma_cut"] = float(np.mean([c["bars_joints"][1][c["k"]] for c in out["cuts"]])) if cuts and closed else np.nan
    return out


# ---------------------------------------------------------------- 37x37 patch grid (scripts/run_features.py)

def patch_edges_ok(G, ok=None, lab=None, cube=None):
    """edges of a 37x37 patch graph whose two patch centres are joined by a straight run of ok pixels (ok = 512-px
    figure mask minus occlusion-edge pixels): at patch resolution the bars touch across the Penrose loop's narrow inner
    hole and across occlusions, and those edges must not carry depth changes. With lab (cube per node) and cube (the
    pixel cube map) the run must also lie on the two patches' own cubes and touch both: a patch's majority cube need
    not be under its centre (C_a090_k1_tau010: a patch of cube 6 by 87 to 85 px whose centre run lies on cube 4 joined
    cubes 6 and 4 across the occlusion at corner 1 and closed the open loop)"""
    r0, c0 = 14 * G["r"][G["p"]] + 7, 14 * G["c"][G["p"]] + 7          # 518-px centre of p; q is 14 px down / right
    good = np.ones(len(G["p"]), bool)
    if cube is not None:
        a, b = lab[G["p"]], lab[G["q"]]
        ta, tb = np.zeros(len(a), bool), np.zeros(len(a), bool)
    for i in range(15):
        rr = np.where(G["ax"] == 0, r0 + i, r0); cc = np.where(G["ax"] == 1, c0 + i, c0)
        if ok is not None:
            good &= ok[IDX[rr], IDX[cc]]
        if cube is not None:
            cu = cube[IDX[rr], IDX[cc]]
            good &= (cu == a) | (cu == b); ta |= cu == a; tb |= cu == b
    return good & ta & tb if cube is not None else good


def patch_readout(g37, maj, keep, px, n_bars=3, ok=None, removed=None, cube=None, w=None, **kw):
    """readout() on the 37x37 patch grid. g37 [37, 37, 2]: per-patch gradient (x right, y up; a probe or head output,
    or the exact 'grad' of a feature file, NaN where unknown), maj: majority cube per patch, keep: patches used
    ([1369] or [37, 37], e.g. the feature file's keep = figure coverage >= 0.25), px: stimulus pixel size (cube units).
    ok: 512-px mask of figure pixels off occlusion edges (exact, or the figure minus predicted edges): only edges whose
    centre-to-centre segment stays on it are used (recommended: without it edges cross the inner hole). removed:
    patches to drop. cube: the image's pixel cube map, for the bar geometry (recommended; without it the geometry comes
    from maj and is off by up to ~8 deg) and to keep only edges whose segment runs on the two patches' own majority
    cubes (patch_edges_ok). w: extra per-edge weights on graph(mask). Depths come out in cube units."""
    mask = np.asarray(keep, bool).reshape(G37, G37) & (maj >= 0) & (True if removed is None else ~removed)
    n_cubes = N_BAR * n_bars
    masks = [mask & (maj == c) for c in range(n_cubes)]
    if cube is not None and "geom" not in kw:
        kw["geom"] = bar_geometry(cube, n_bars, px)
    if ok is not None or cube is not None:
        G = graph(mask)
        good = patch_edges_ok(G, ok, np.asarray(maj)[G["r"], G["c"]], cube).astype(float)
        w = good if w is None else good * w
    return readout(g37, mask, np.asarray(maj), PATCH * px, n_bars, w=w, masks=masks, **kw)


def patch_profile(d37, maj, keep, n_cubes):
    """per-cube median of a per-patch map (e.g. the feature file's dgrid[k], the exact depth of object k)"""
    m = np.asarray(keep, bool).reshape(G37, G37)
    return np.array([np.nanmedian(d37[m & (maj == c)]) if (m & (maj == c)).any() else np.nan for c in range(n_cubes)])


# ---------------------------------------------------------------- exact data (stimuli, rings)

def pixel_size(st):
    """cube units per pixel: the ring's own px; for a stimulus, from its corner_xy (a bar's image is N_BAR sqrt(2/3)
    cube units long): pdepth.normals.PX (to 1e-15) except set D's zoomed 'small' images, PX / 0.7; else PX"""
    if "px" in st:
        return float(st["px"])
    if "corner_xy" in st:
        c = np.asarray(st["corner_xy"], float)
        half = st["fg"].shape[1] / 2 if "fg" in st else 256
        return float(N_BAR * np.sqrt(2 / 3) / (np.linalg.norm(np.roll(c, -1, 0) - c, axis=1).mean() * half))
    from pdepth.normals import PX
    return PX


def exact_g(st, px=None):
    """exact gradient [H, W, 2] on the whole figure: pdepth.normals.gt_normals on face interiors (identical for the
    three objects and the impossible image); every other figure pixel takes its nearest face-interior value"""
    from pdepth.normals import gt_normals
    n, ok = gt_normals(st, pixel_size(st) if px is None else px)
    g = np.stack([n[..., 0] / n[..., 2], n[..., 1] / n[..., 2]], -1)
    i = distance_transform_edt(~ok, return_distances=False, return_indices=True)
    g = g[i[0], i[1]]
    g[~st["fg"]] = np.nan
    return g


def occlusion_edges(st, px=None, tau_px=4.0):
    """figure pixels on either side of a depth jump larger than tau_px crease steps (pdepth/scenes.py's rule) in every
    depth map the image is an image of: the impossible image, its three objects (the joints it draws are real in two of
    them, so only true occlusions remain); a possible control, its own object (also set C at tau = 0, which shows the
    impossible image: its gap is then an edge although nothing in the image marks it); a ring, its only map"""
    D = st["depth"].astype(float)
    if int(st["possible"]) >= 0:
        D = D[[int(st["possible"])]]
    tau = tau_px * (pixel_size(st) if px is None else px) * np.sqrt(2)
    with np.errstate(invalid="ignore"):
        jh = (np.abs(np.diff(D, axis=2)) > tau).all(0)               # NaN (background) never counts as a jump
        jv = (np.abs(np.diff(D, axis=1)) > tau).all(0)
    e = np.zeros(st["fg"].shape, bool)
    e[:, 1:] |= jh; e[:, :-1] |= jh; e[1:] |= jv; e[:-1] |= jv
    return e & st["fg"]
