"""Figure 2: the networks bend the impossible loop, opening it removes the bend, the features keep it,
and a head that outputs depth changes keeps it too.
(a) total excess tear sum_j e_j on the intact impossible image (set A) per network, median and 95% cluster-bootstrap
    interval over the layout bins (results/summary_excess.json, samplers: all noise seeds pooled; falls back to
    summary_probe.json). The mean of the three objects and every single object give 1 (pink band, its note in the
    free top left corner); the ordinary step at every junction gives 0 (dotted line, its note in the free bottom
    left corner, under every interval). Network names stand upright (90 degrees) in (a), (b) and (c) right.
(b) erase one junction of the loop: the growth of the jump there (the "released" jump of scripts/analyze_loops.py), in
    units of the output's own mean bar change (median and IQR over images x junctions, results/summary_loops.json).
    The Penrose loop's mismatch T is three bar changes (dashed line); a closed possible ring (skew hexagon, rhombus)
    has none. Right of the thin line: the matched depth (0-form) head on frozen DA-S, two seeds.
(c) left: layer-wise rho = bar slope on the impossible image / bar slope on the matched controls (median, 95% CI) for
    the ridge probes trained on random possible scenes only (results/summary_layers_<net>.json; decodable layers only,
    L6-L12 encoder blocks, F1-F4 decoder stages, out = 0-form probe on the network's output, star): one line per probe
    and network, depth-change (1-form) probe teal, depth (0-form) probe brown, DA-S solid line and filled star,
    DA-B dashed line and hollow star (drawn only if its summary exists); the lines' intervals are drawn light, every
    layer has a tick and every second encoder block a label. Both depth probes are drawn from L8: at L6-L7 they read
    even the controls' bar slope at under 0.04 of exact (raw, before calibration), so rho there is a ratio over ~0
    (printed with that control slope under "c skipped"). The caption's bound for every probe before the first layer
    drawn (under 0.15 of the controls' slope, raw) is printed by scripts/paper_numbers.py. An interval leaving the axis
    ends in a small caret. +1 = bars as on the matched control, -1 = reversed.
    right (titled "outputs", the layer plot's title "(c) layer probes" sits over the left part only): the same
    ratio (relative tilt) for three surface-normal networks, each with its depth twin on the same
    backbone (results/summary_normals.json): one group per network, Marigold v1.1, Lotus-D, E2E-FT from left to
    right, normals (teal diamond) left, depth twin (brown diamond) right, a grey line joining the pair. E2E-FT's depth
    interval runs past the top of the axis (caret), so it is the last group and the key sits top left, above the data.
(d) matched heads on the same frozen DA-S features and the same possible-only training scenes
    (results/summary_heads.json, two seeds): bar slope / exact on impossible images and controls, the loop integral
    Gamma / T on impossible images and closed rings (exact 0), and the share of cuts at junction k that return object
    k; depth-change head = 1-form head (teal), depth head = 0-form head (brown). Bar = mean of the two seeds, dots =
    the seeds (with their 95% CI where the summary has one). The row names sit left of the bars; every bar's seed mean
    is written at its end (a bar that reaches the exact line has its value just right of that line). For a depth map
    every cut of the loop gives the same profile, so the depth head's "cut at k gives object k" is 1/3 by
    construction (written as 1/3).
Print size: the canvas is 7.0 x 2.25 in and is printed at \\textwidth (7.06 in), so every text is at least
FS * 7.06 / 7.0 = 7.06 pt. The run prints the smallest size and a layout check: texts leaving the canvas, overlapping
texts (rotated tick labels by their rotated outlines; neighbouring rotated tick labels of one axis by the boxes around
their glyphs, with the smallest gap printed), texts or legend boxes over data marks, and texts over the reference lines
(dashed exact lines, zero lines, separators).
usage: python figures/fig_main.py   -> figures/out/fig_main.{pdf,png} (CPU, from results/ only)"""
import os, sys, json, itertools, math, matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from style import MODELS, setup, short

setup()
W, H = 7.0, 2.25                             # inches; printed at \textwidth (7.06 in)
PRINT_W = 7.06
FS = 7.0                                     # smallest font in the figure (pt on the canvas, 7.06 pt printed)
LS = 7.5                                     # axis labels
plt.rcParams.update({"savefig.bbox": "standard",   # exact W x H canvas; the layout below keeps every label inside
                     "axes.titlesize": 8.5, "axes.titlepad": 3.0, "axes.labelsize": LS, "legend.fontsize": FS,
                     "xtick.labelsize": FS, "ytick.labelsize": FS, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                     "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
ORDER = ["dav2-s", "dav2-b", "dav2-l", "dpt-l", "zoe", "lotus-d", "lotus-g", "e2eft", "mg11-1", "mg10-1"]
MODELS["mg11-1"] = dict(label="Marigold v1.1 (1 step)", color="#c51b7d")
MODELS["mg10-1"] = dict(label="Marigold v1.0 (1 step)", color="#f768a1")
MARK = {"dav2-s": "o", "dav2-b": "s", "dav2-l": "D", "dpt-l": "^", "zoe": "v", "lotus-d": "o", "lotus-g": "s",
        "e2eft": "D", "mg11-1": "^", "mg10-1": "v"}
ONE, ZERO = "#01665e", "#8c510a"            # 1-form (depth changes, normals) vs 0-form (depth), panels (b)-(d)
ONE_L, ZERO_L = "#9fd3cc", "#e3c38f"        # their light versions (bars in (d))
REF_OBJ, RING, INK = "#b2182b", "0.45", "0.2"
TICK = dict(rotation=90, ha="right", va="center_baseline", rotation_mode="anchor", fontsize=FS)   # upright names
MINUS = chr(0x2212)


def load(name):
    p = os.path.join(ROOT, "results", name)
    return json.load(open(p)) if os.path.exists(p) else None


def note(ax, x, y, s, color="0.3", ha="right", va="center", **kw):
    """a note in free space (no white box: the layout check below shows it lies on no mark, line or spine)"""
    ax.text(x, y, s, ha=ha, va=va, fontsize=FS, color=color, zorder=4, **kw)


def ref(artist_or_list):
    """tag reference lines (exact values, zero lines, separators): the layout check reports texts over them apart
    from texts over data"""
    for a in (artist_or_list if isinstance(artist_or_list, (list, tuple)) else [artist_or_list]):
        a.set_gid("ref")
    return artist_or_list


def err(ax, x, v, lo, hi, ylim, color, fmt="o", mfc=None, ms=3.4, lw=0.9, z=3, vlab=(0.3, 0.0, "left"), ecolor=None,
        alpha=1.0):
    """median with interval; an interval leaving the axis is cut at the edge and marked with a caret; a median
    outside the axis is drawn as a caret at the edge with its value written next to it"""
    b0, b1 = ylim
    pad = 0.012 * (b1 - b0)
    lo_c, hi_c = max(lo, b0), min(hi, b1)
    ec = ecolor or color
    if b0 <= v <= b1:
        ax.vlines(x, lo_c, hi_c, color=ec, lw=lw, zorder=z, alpha=alpha)
        if fmt:
            ax.plot(x, v, marker=fmt, color=color, mfc=mfc or color, ms=ms, mew=0.8, zorder=z + 0.5, ls="")
    else:
        up = v > b1
        edge = b1 - 2 * pad if up else b0 + 2 * pad
        if hi_c > lo_c:
            ax.vlines(x, lo_c, hi_c, color=ec, lw=lw, zorder=z, alpha=alpha)
        ax.plot(x, edge, marker="^" if up else "v", color=color, mfc=mfc or color, ms=ms, mew=0.8, zorder=z + 1,
                clip_on=False)
        ax.text(x + vlab[0], edge + vlab[1], f"{v:+.1f}", ha=vlab[2], va="center", fontsize=FS, color="0.25",
                zorder=z + 1, bbox=dict(fc="white", ec="none", pad=0.15))
        return
    if hi > b1:
        ax.plot(x, b1 - pad, marker="^", color=ec, ms=2.6, mew=0, zorder=z, clip_on=False)
    if lo < b0:
        ax.plot(x, b0 + pad, marker="v", color=ec, ms=2.6, mew=0, zorder=z, clip_on=False)


def axes_in(fig, x, y, w, h, **kw):
    """axes placed in inches from the figure's lower left corner"""
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


# ---------------------------------------------------------------- layout check
def _outline(t, r):
    """the four corners of a text's (possibly rotated) box in display pixels"""
    bb = t.get_window_extent(r)
    rot = t.get_rotation() % 360
    if not rot:
        return np.array([(bb.x0, bb.y0), (bb.x1, bb.y0), (bb.x1, bb.y1), (bb.x0, bb.y1)])
    t.set_rotation(0)
    b0 = t.get_window_extent(r)
    t.set_rotation(rot)
    c, (w, h), a = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2]), (b0.width, b0.height), math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return c + np.array([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]) @ R.T


def _ink(t, r):
    """the (possibly rotated) box around a one-line text's glyphs (exact outlines, not the line box) in display
    pixels; used for the gap between neighbouring rotated tick labels, whose line boxes may touch"""
    fs = t.get_fontsize() * t.figure.dpi / 72
    ext = TextPath((0, 0), t.get_text(), size=fs, prop=t.get_fontproperties()).get_extents()
    _, _, lp_d = r.get_text_width_height_descent("lp", t.get_fontproperties(), ismath=False)
    _, _, d = r.get_text_width_height_descent(t.get_text(), t.get_fontproperties(), ismath=False)
    bb = t.get_window_extent(r)
    rot = t.get_rotation() % 360
    t.set_rotation(0)
    b0 = t.get_window_extent(r)
    t.set_rotation(rot)
    c, a = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2]), math.radians(rot)
    yb = -b0.height / 2 + max(d, lp_d)                  # baseline, from the box centre
    x0 = -b0.width / 2
    P = np.array([(x0 + ext.x0, yb + ext.y0), (x0 + ext.x1, yb + ext.y0), (x0 + ext.x1, yb + ext.y1),
                  (x0 + ext.x0, yb + ext.y1)])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return c + P @ R.T


def _gap(p, q):
    """the largest gap between convex polygons p, q along their edge normals (pixels; negative: they overlap)"""
    best = -np.inf
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / (np.hypot(*e) + 1e-12)
            a, b = p @ n, q @ n
            best = max(best, b.min() - a.max(), a.min() - b.max())
    return best


def _sep(p, q, tol=0.5):
    """True if convex polygons p, q are separated by more than -tol pixels on some edge normal (SAT)"""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / (np.hypot(*e) + 1e-12)
            a, b = p @ n, q @ n
            if a.max() <= b.min() + tol or b.max() <= a.min() + tol:
                return True
    return False


def _marks(fig, r):
    """data and reference marks of every axes as (points [N,2] in pixels, radius px, kind, axes index, name)"""
    out = []
    for k, ax in enumerate(fig.axes):
        abb = ax.get_window_extent(r)

        def keep(P, art):
            if art.get_clip_on():
                P = P[(P[:, 0] >= abb.x0 - 0.5) & (P[:, 0] <= abb.x1 + 0.5) & (P[:, 1] >= abb.y0 - 0.5)
                      & (P[:, 1] <= abb.y1 + 0.5)]
            return P

        def sample(P):
            pts = [P[:1]]
            for a, b in zip(P[:-1], P[1:]):
                n = max(int(np.hypot(*(b - a))), 1)
                pts.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
            return np.concatenate(pts)

        for ln in ax.lines:
            if not ln.get_visible():
                continue
            kind = "ref" if ln.get_gid() == "ref" else "data"
            P = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
            P = P[np.isfinite(P).all(1)]
            if len(P) > 1 and ln.get_linestyle() not in ("None", "", " "):
                out.append((keep(sample(P), ln), ln.get_linewidth() * fig.dpi / 144, kind, k, "line"))
            if ln.get_marker() not in (None, "None", "", " "):
                out.append((keep(P, ln), ln.get_markersize() * fig.dpi / 144, kind, k, "marker"))
        for co in ax.collections:
            if isinstance(co, LineCollection) and co.get_visible():
                kind = "ref" if co.get_gid() == "ref" else "data"
                for seg in co.get_segments():
                    P = co.get_transform().transform(np.asarray(seg, float))
                    out.append((keep(sample(P), co), co.get_linewidth()[0] * fig.dpi / 144, kind, k, "interval"))
        for pa in ax.patches:
            if pa.get_visible():
                bb = pa.get_window_extent(r)
                if bb.width > 0.3 and bb.height > 0.3:
                    xs, ys = np.meshgrid(np.linspace(bb.x0, bb.x1, max(int(bb.width), 2)),
                                         np.linspace(bb.y0, bb.y1, max(int(bb.height), 2)))
                    kind = "ref" if pa.get_gid() == "ref" else "data"
                    out.append((keep(np.c_[xs.ravel(), ys.ravel()], pa), 0.0, kind, k, "bar"))
        for name, sp in ax.spines.items():
            if sp.get_visible():
                P = sp.get_transform().transform(sp.get_path().vertices)
                out.append((sample(P), sp.get_linewidth() * fig.dpi / 144, "spine", k, f"{name} spine"))
    return out


def layout_check(fig):
    """smallest printed font size, texts leaving the canvas, overlapping texts (rotated texts by their rotated
    outlines), and texts / legend boxes lying on data marks, on reference lines or on the axes spines"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = PRINT_W / W
    tick_axis = {}                                      # tick label -> the axis that owns it (t.axes is not set)
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            for t in axis.get_ticklabels():
                tick_axis[id(t)] = axis
    items = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        axis = tick_axis.get(id(t))
        if axis is not None:
            lo, hi = sorted(axis.get_view_interval())
            loc = t.get_position()[0 if axis.axis_name == "x" else 1]
            if not lo - 1e-9 <= loc <= hi + 1e-9:
                continue                                # tick outside the view: not drawn
        bb = t.get_window_extent(r)
        if bb.width > 0.5:
            items.append((t, bb, _outline(t, r)))
    small = min(t.get_fontsize() for t, _, _ in items) * k
    out = [t.get_text() for t, b, _ in items if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W * fig.dpi + 0.5
           or b.y1 > H * fig.dpi + 0.5]

    def neighbour_ticks(t1, t2):    # rotated tick labels of one axis: their boxes (line height) touch diagonally
        a1, a2 = tick_axis.get(id(t1)), tick_axis.get(id(t2))   # while the glyphs do not; checked against the rest
        return a1 is not None and a1 is a2 and t1.get_rotation() and t2.get_rotation()
    over = [(t1.get_text(), t2.get_text()) for (t1, _, p1), (t2, _, p2) in itertools.combinations(items, 2)
            if not neighbour_ticks(t1, t2) and not _sep(p1, p2)]
    # rotated tick labels of one axis: the gap between neighbours' glyph boxes (pt, printed); under 0 they overlap
    tick_gap = {}
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            labs = [t for t, _, _ in items if tick_axis.get(id(t)) is axis and t.get_rotation()]
            labs.sort(key=lambda t: t.get_position()[0 if axis.axis_name == "x" else 1])
            for t1, t2 in zip(labs[:-1], labs[1:]):
                g = _gap(_ink(t1, r), _ink(t2, r)) * 72 / fig.dpi * k
                tick_gap[f"{t1.get_text()} | {t2.get_text()}"] = round(g, 2)
    if tick_gap:
        low = min(tick_gap, key=tick_gap.get)
        over += [(s, "glyph gap %.2f pt" % g) for s, g in tick_gap.items() if g < 0.5]
    marks = _marks(fig, r)
    axis_texts = {id(a.xaxis.label) for a in fig.axes} | {id(a.yaxis.label) for a in fig.axes} | set(tick_axis)
    boxes = [(t.get_text(), p, id(t) in axis_texts) for t, _, p in items]
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            bb = lg.get_window_extent(r)
            boxes.append(("legend of axes %d" % fig.axes.index(ax),
                          np.array([(bb.x0, bb.y0), (bb.x1, bb.y0), (bb.x1, bb.y1), (bb.x0, bb.y1)]), False))
    on_data, on_ref = [], []
    for name, p, axis_text in boxes:
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        for P, rad, kind, ka, what in marks:
            if not len(P) or (kind == "spine" and axis_text):   # tick and axis labels sit by their spines
                continue
            m = (P[:, 0] > x0 - rad + 0.5) & (P[:, 0] < x1 + rad - 0.5) & (P[:, 1] > y0 - rad + 0.5) & (
                P[:, 1] < y1 + rad - 0.5)
            if not m.any():
                continue
            hit = any(not _sep(p, q + np.array([(-rad, -rad), (rad, -rad), (rad, rad), (-rad, rad)]) * 1.0)
                      for q in P[m][:: max(1, int(m.sum() // 200))][:, None, :])
            if hit:
                (on_data if kind == "data" else on_ref).append((name, f"axes {ka} {what}"))
    uniq = lambda L: sorted(set(L))
    return dict(smallest_font_pt_printed=round(small, 2), outside_canvas=out, overlapping_texts=over,
                smallest_glyph_gap_rotated_ticks_pt=[low, tick_gap[low]] if tick_gap else None,
                text_or_legend_on_data=uniq(on_data), text_on_reference_lines_or_spines=uniq(on_ref))


if __name__ == "__main__":
    EX, PR, LO = load("summary_excess.json"), load("summary_probe.json"), load("summary_loops.json")
    NO, HE = load("summary_normals.json"), load("summary_heads.json")
    LAY = {n: load(f"summary_layers_{n}.json") for n in ("dav2-s", "dav2-b")}
    LAY = {n: v for n, v in LAY.items() if v}
    models = [m for m in ORDER if m in (EX or {}) or m in (PR or {})]
    fig = plt.figure(figsize=(W, H))
    Y0, AH = 0.64, 1.48                          # axes bottom and height (in), shared by all panels
    axes = [axes_in(fig, 0.40, Y0, 0.98, AH),    # (a)
            axes_in(fig, 1.74, Y0, 1.21, AH),    # (b)
            axes_in(fig, 3.40, Y0, 1.25, AH),    # (c) layers
            None,                                # (c) normals vs depth twins, added below (shares y)
            axes_in(fig, 5.99, Y0, 0.97, AH)]    # (d); its row names sit in the 0.5 in left of it
    axes[3] = axes_in(fig, 4.71, Y0, 0.69, AH, sharey=axes[2])
    printed = {}

    # (a) intact image: total excess tear on the intact impossible image
    ax = axes[0]
    ylim = (-0.8, 1.15)
    for i, m in enumerate(models):
        src = EX[m]["A"]["sum_e"] if EX and m in EX else PR[m]["impossible_intact"]["sum_e"]
        v, lo, hi = src
        err(ax, i, v, lo, hi, ylim, MODELS[m]["color"], fmt=MARK[m], ms=3.6, lw=1.0)
        printed.setdefault("a", {})[m] = (round(v, 2), round(lo, 2), round(hi, 2))
    n = len(models)
    ref(ax.axhspan(0.97, 1.03, color="#e34a33", alpha=0.25, lw=0))
    # the two notes sit in the free corners: top left under the band, bottom left under every interval
    note(ax, -0.4, 0.93, "one object\nor a mean: 1", color=REF_OBJ, ha="left", va="top", linespacing=0.95)
    ref(ax.axhline(0, color="0.4", lw=0.7, ls=":"))
    note(ax, -0.4, -0.75, "ordinary step\nat every junction: 0", ha="left", va="bottom", linespacing=0.95)
    ax.set_xticks(range(n)); ax.set_xticklabels([short(m) for m in models], **TICK)
    ax.set_xlim(-0.55, n - 0.45); ax.set_ylim(*ylim)
    ax.set_yticks([-0.5, 0, 0.5, 1]); ax.set_yticklabels([MINUS + "0.5", "0", "0.5", "1"])
    ax.set_ylabel(r"total excess tear $\Sigma_j\, e_j$", labelpad=1)
    ax.set_title("(a) intact image")

    # (b) erased junction: jump released at the erased junction, Penrose loop vs closed possible rings
    ax = axes[1]
    ylim = (-1.75, 8.9)
    heads0 = [h for h in ("head-0form-s-seed0", "head-0form-s-seed1") if h in LO]
    xs = list(range(len(models))) + [len(models) + 0.5 + i for i in range(len(heads0))]
    keys = models + heads0
    for x, m in zip(xs, keys):
        for dx, s in [(-0.27, "A"), (0.0, "H"), (0.27, "R")]:
            r = LO[m][s]
            v, (lo, hi) = r["released"], r["released_iqr"]
            if s == "A":
                col = MODELS[m]["color"] if m in MODELS else ZERO
                err(ax, x + dx, v, lo, hi, ylim, col, fmt=MARK.get(m, "o"), ms=3.4, lw=1.0, z=4)
            else:
                err(ax, x + dx, v, lo, hi, ylim, RING, fmt="h" if s == "H" else "d", mfc="white", ms=3.0, lw=0.7)
            printed.setdefault("b", {}).setdefault(m, {})[s] = (round(v, 2), round(lo, 2), round(hi, 2))
    ref(ax.axhline(3, color=REF_OBJ, lw=0.7, ls="--", zorder=1))
    ref(ax.axhline(0, color="0.4", lw=0.7, ls=":", zorder=1))
    if heads0:
        ref(ax.axvline(len(models) - 0.2, color="0.75", lw=0.6, ymax=0.68))
    ax.set_xticks(xs)
    ax.set_xticklabels([short(m) for m in models] + [f"depth head s{h[-1]}" for h in heads0], **TICK)
    ax.set_xlim(-0.6, xs[-1] + 0.5); ax.set_ylim(*ylim)
    ax.set_yticks([0, 3, 6])
    ax.set_ylabel("growth of the jump at the\nerased junction (bar changes)", labelpad=1.5, linespacing=0.95)
    ax.set_title("(b) erased junction")
    hd = [Line2D([], [], color="0.25", marker="o", ls="", ms=3.4, label="Penrose loop"),
          Line2D([], [], color=RING, mfc="white", marker="h", ls="", ms=3.2, label="hexagon ring"),
          Line2D([], [], color=REF_OBJ, ls="--", lw=0.7, label="exact $T$ = 3"),
          Line2D([], [], color=RING, mfc="white", marker="d", ls="", ms=3.2, label="rhombus ring")]
    ax.legend(handles=hd, frameon=False, loc="upper left", ncol=2, handletextpad=0.25, borderaxespad=0.0,
              labelspacing=0.25, columnspacing=0.6, handlelength=1.0, bbox_to_anchor=(0.0, 1.0))

    # (c) layer probes: rho per layer, one line per probe (1-form teal, 0-form brown) and network
    #     (DA-S solid, DA-B dashed); the probe on the network's output (out) is a star (DA-S filled, DA-B hollow)
    ax, axn = axes[2], axes[3]
    ylim = (-2.05, 2.08)
    LAYERS = [f"L{i}" for i in range(6, 13)] + ["F1", "F2", "F3", "F4"]
    X = {l: i for i, l in enumerate(LAYERS[:7])}
    X.update({f"F{i}": 6.4 + i for i in range(1, 5)})
    X["OUT"] = X["F4"] + 1.45
    NET = {"dav2-s": dict(ls="-", filled=True, dx=-0.11), "dav2-b": dict(ls="--", filled=False, dx=0.11)}
    # depth (0-form) probes from L8: at L6-L7 they miss even the controls' bar slope (raw slope / exact on the controls
    # B: DA-S 0.02, -0.00; DA-B -0.02, 0.03), so rho is a ratio over ~0 (DA-S -0.97, +0.33 with CI [-2.25, 2.34] at
    # L7; DA-B +3.4, -2.4); printed under "c skipped"; the caption's bound: scripts/paper_numbers.py
    SKIP = {("dav2-s", "zeroform"): ("L6", "L7"), ("dav2-b", "zeroform"): ("L6", "L7")}
    for y, c, ls in [(1, REF_OBJ, "--"), (-1, "0.35", "--"), (0, "0.7", ":")]:
        for a in (ax, axn):
            ref(a.axhline(y, color=c, lw=0.7, ls=ls, zorder=1))
    for xb in (X["L12"] + 0.7, X["F4"] + 0.72):          # stops below the "+1" note
        ref(ax.axvline(xb, color="0.85", lw=0.5, zorder=0, ymax=(1.5 - ylim[0]) / (ylim[1] - ylim[0])))
    for net, S in LAY.items():
        st = NET[net]
        for probe, col in [("oneform", ONE), ("zeroform", ZERO)]:
            mfc = col if st["filled"] else "white"
            line = []
            for l in LAYERS + ["OUT"]:
                p = S["layers"].get(l, {}).get(probe)
                if not p or not p.get("decodable") or p["A"].get("rho") is None:
                    continue
                if (net, probe) in SKIP and l in SKIP[(net, probe)]:
                    printed.setdefault("c skipped", {}).setdefault(f"{net} {probe}", {})[l] = dict(
                        rho=round(p["A"]["rho"], 2), control_slope_over_exact=round(p["B"]["ratio"], 3))
                    continue
                v, (lo, hi) = p["A"]["rho"], p["A"]["rho_ci"]
                x = X[l] + st["dx"]
                if l == "OUT":
                    err(ax, x, v, lo, hi, ylim, col, fmt="*", mfc=mfc, ms=5.4, lw=0.7, z=4)
                else:
                    err(ax, x, v, lo, hi, ylim, col, fmt=None, lw=0.55, z=2, alpha=0.45)
                    line.append((x, v))
                printed.setdefault("c", {}).setdefault(f"{net} {probe}", {})[l] = (round(v, 2), round(lo, 2), round(hi, 2))
            if line:
                ax.plot(*zip(*line), color=col, lw=1.1 if st["filled"] else 1.0, ls=st["ls"], zorder=3)
    ax.set_xticks([X[l] for l in LAYERS] + [X["OUT"]])
    ax.set_xticklabels(["L6", "", "8", "", "10", "", "12", "F1", "2", "3", "4", "out"])
    ax.set_xlim(-0.55, X["OUT"] + 0.55); ax.set_ylim(*ylim)
    ax.set_yticks([-2, -1, 0, 1]); ax.set_yticklabels([MINUS + "2", MINUS + "1", "0", "+1"])
    for xc, s in [((X["L6"] + X["L12"]) / 2, "encoder blocks"), ((X["F1"] + X["F4"]) / 2, "decoder")]:
        ax.annotate(s, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -11.5), textcoords="offset points",
                    ha="center", va="top", fontsize=FS, color="0.35")
    note(ax, X["L8"] - 0.1, 1.72, "+1: as on the control", color=REF_OBJ, ha="left")
    note(ax, X["L6"] - 0.35, -1.36, MINUS + "1: reversed", ha="left")
    ax.set_ylabel("$\\rho$ = bar slope ratio,\nimpossible / control", labelpad=1, linespacing=0.95)
    hd = [Line2D([], [], color=ONE, lw=1.0, label="depth-change probe"),
          Line2D([], [], color=ZERO, lw=1.0, label="depth probe"),
          Line2D([], [], color="0.3", lw=1.1, marker="*", ms=5.0, mew=0.8, label="DA-S"),
          Line2D([], [], color="0.3", lw=1.0, ls="--", marker="*", ms=5.0, mfc="white", mew=0.8, label="DA-B")]
    ax.legend(handles=hd if len(LAY) > 1 else hd[:3], frameon=False, ncol=2, loc="upper left",
              bbox_to_anchor=(-0.27, -0.205), handlelength=1.6, handletextpad=0.4, columnspacing=0.9,
              labelspacing=0.25, borderaxespad=0.0)
    # right: surface-normal networks vs their depth twins on the same backbones, one group per network (normals
    # left, depth right); E2E-FT last: its depth interval leaves the top of the axis, the key sits top left
    NPAIR = [("mg11-n", "mg11-1"), ("lotus-d-n", "lotus-d"), ("e2eft-n", "e2eft")]
    DXN = 0.2
    shown = []
    for g, (k, m) in enumerate([p for p in NPAIR if NO and p[0] in NO]):
        v, lo, hi = NO[k]["rel_tilt"]
        dv, dlo, dhi = NO[k]["depth"]["rel_tilt"]
        axn.plot([g - DXN, g + DXN], [v, dv], color="0.75", lw=0.6, zorder=1)
        err(axn, g - DXN, v, lo, hi, ylim, ONE, fmt="D", ms=3.0, lw=0.75)
        err(axn, g + DXN, dv, dlo, dhi, ylim, ZERO, fmt="D", ms=3.0, lw=0.75)
        shown.append(m)
        printed.setdefault("c normals", {})[k] = dict(normals=(round(v, 2), round(lo, 2), round(hi, 2)),
                                                      depth=(round(dv, 2), round(dlo, 2), round(dhi, 2)))
    axn.set_xlim(-0.55, len(shown) - 0.45)
    axn.set_xticks(range(len(shown))); axn.set_xticklabels([short(m) for m in shown], **TICK)
    axn.tick_params(axis="y", left=False, labelleft=False)
    axn.spines["left"].set_color("0.75")
    hd = [Line2D([], [], color=ONE, marker="D", ls="", ms=3.0, label="normals"),
          Line2D([], [], color=ZERO, marker="D", ls="", ms=3.0, label="depth twin")]
    axn.legend(handles=hd, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=0.8,
               handletextpad=0.3, labelspacing=0.2, borderaxespad=0.0, borderpad=0.0)
    ax.set_title("(c) layer probes")
    axn.set_title("outputs")

    # (d) matched heads on the same frozen features (bar: mean of two seeds, dots: the seeds)
    ax = axes[4]
    ROWS = [("slope", "A", "bar slope,\nimpossible", 0.0), ("slope", "B", "bar slope,\ncontrols", 1.0),
            ("gamma", "A", "$\\Gamma/T$,\nimpossible", 2.25), ("gamma", "rings", "$\\Gamma/T$, rings\n(exact 0)", 3.25),
            ("cut", None, "cut at $k$ gives\nobject $k$", 4.5)]
    HEADS = [("1form", ONE, ONE_L, -0.19), ("0form", ZERO, ZERO_L, 0.19)]   # y offset of the bar inside the row
    BH = 0.27
    for q, s, lab, yr in ROWS:
        for kind, col, light, dy in HEADS:
            vals, ends = [], []
            for seed in (0, 1):
                h = f"{kind}-s-seed{seed}"
                if h not in HE:
                    continue
                r = HE[h]
                val = r["slope_ratio"][s] if q == "slope" else (r["gamma_bars"][s] if q == "gamma" else r["cut_gives_object"])
                v, lo, hi = val if isinstance(val, list) else (val, val, val)
                vals.append(v); ends.append(hi)
                if hi > lo:
                    ax.hlines(yr + dy, lo, hi, color=col, lw=0.7, zorder=3)
                ax.plot(v, yr + dy, marker="o", ms=2.7, color=col, mec="white", mew=0.4, ls="", zorder=4,
                        clip_on=False)
                printed.setdefault("d", {}).setdefault(h, {})[lab.replace("\n", " ")] = tuple(
                    round(x, 3) for x in (v, lo, hi))
            if vals:
                m = sum(vals) / len(vals)
                ax.barh(yr + dy, m, height=BH, color=light, lw=0, zorder=2)
                printed.setdefault("d bar (seed mean)", {}).setdefault(kind, {})[lab.replace("\n", " ")] = round(m, 3)
                # the seed mean at the bar's end; a bar reaching the exact line has it just right of the line, and a
                # depth map gives every cut of the loop the same profile, so 1/3 there is exact (by construction)
                txt = "1/3" if (q == "cut" and abs(m - 1 / 3) < 1e-6) else f"{m:.2f}".replace("-", MINUS)
                xe = max(ends + [m, 0.0])
                xt = 1.035 if xe > 0.9 else xe + 0.035
                ax.text(xt, yr + dy, txt, ha="left", va="center", fontsize=FS, color=INK, zorder=4)
    ref(ax.axvline(1, color=REF_OBJ, lw=0.7, ls="--", zorder=1))
    ax.set_ylim(ROWS[-1][3] + 0.5, -0.5)
    ax.set_xlim(0, 1.27)
    ax.set_yticks([yr for *_, yr in ROWS]); ax.set_yticklabels([lab for _, _, lab, _ in ROWS])
    for t in ax.get_yticklabels():
        t.set_linespacing(0.95)
    ax.tick_params(axis="y", left=False, pad=2.5)
    ax.spines["left"].set_color("0.4")
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("ratio to exact, or share", labelpad=1, x=1 / 1.27 / 2)
    ax.set_title("(d) matched heads", x=0.24)
    hd = [Patch(fc=ONE_L, ec="none", label="depth-change head"), Patch(fc=ZERO_L, ec="none", label="depth head"),
          Line2D([], [], color="0.3", marker="o", ms=2.7, mec="white", mew=0.4, ls="", label="seeds 0, 1"),
          Line2D([], [], color=REF_OBJ, lw=0.7, ls="--", label="exact")]
    ax.legend(handles=hd, ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(1.0, -0.205),
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.0)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    printed["layout"] = dict(figure_in=(W, H), tight_bbox_in=[round(t, 3) for t in tb.bounds], **layout_check(fig))
    fig.savefig(os.path.join(HERE, "out", "fig_main.pdf"), bbox_inches=None)
    fig.savefig(os.path.join(HERE, "out", "fig_main.png"), dpi=220, bbox_inches=None)
    print(json.dumps(printed, indent=1, ensure_ascii=False))
