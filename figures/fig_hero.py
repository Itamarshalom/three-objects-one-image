"""Figure 1: three objects that give one image, DA-S's depth on it, and a depth-change head summed around the loop.

Top: each picture sits above the bottom panel that plots it. Over (i): the three trimmed objects 12 deg off the
special view (side views cached in figures/out/side_trimmed_k.png, rendered for layout 90 and turned to this layout
here; delete them to re-render with pdepth.render.render_side_trimmed, a few CPU minutes each). Over (ii): the shared
impossible image (junctions J1-J3 marked) and Depth Anything V2-S's depth on it. Over (iii): what the depth-change
(1-form) head reads on the same frozen DA-S features (trained on random possible scenes only; its output on this image,
written by scripts/run_heads.py to outputs/heads_out/1form-s-seed0, is kept on the figure's pixels in
results/fig1_head-1form-s-seed0_A_a060_p0_t0.npz, so no GPU run is needed; when that file is missing it is written here
from the run_heads.py output): its depth changes summed along the closed pixel path once around the loop
(pdepth.oneform.loop_path, on the figure minus the head's own edge pixels; black, arrows: direction of travel), from its
first node at J1 (left, the same sum as in (iii) below) and from its first node in corner cube 5 at J2 (right, the same
walk started there), spread to every pixel of the figure from the nearest path node of the same cube and drawn with the
DA-S map's colour map (red: near).
The walk's first cube holds both of its ends, so the sum meets itself there with a seam of one lap (0.94 T), which no
depth map can have; started at J2, the seam moves to J2 (a cut at J_k gives object k). DA-S is drawn because the heads
use its frozen features.
Bottom: the loop walked once, from cube 0 (at J1) through cubes 1..14 and back to cube 0; depth in units of the
obstruction T = 5 sqrt3 (up = farther), every curve starting at 0 (cube profiles: bar 1's fitted line at J1, so one
lap ends where bar 3's line ends and the jump back at J1 is its distance to 0; the head: the first node of the walk).
 (i)   object 1's exact per-cube depth (pdepth.readout.cube_profile; lines fitted per bar as in pdepth.loop): nearer
       along each bar, back by T at its gap (J1, the end of the walk).
 (ii)  DA-S's per-cube depth on the impossible image (blue) and on control 1 (grey: object 1 untrimmed, its gap at J1
       visible), scaled so that the jump DA-S puts at control 1's gap (J1) is 1 (pdepth.loop jump; the counterpart of
       T, so the y axis is shared and the grey "tear" arrow is exactly 1, as the +T arrow in (i)). Its excess tears
       e_j on the impossible image (scripts/analyze_excess.py: 0 = the step DA-S puts at an ordinary junction, 1 = its
       jump at a real gap) are printed. The control shows DA-S's ordinary steps at J2 and J3 (both toward the viewer)
       and its tear at J1.
 (iii) the head's depth changes summed along the path (no cut anywhere: pdepth.oneform.edge_form, the same sum as
       circulation_path); grey dashed: the exact depth changes summed along the same path (ends at -T). The arrow is
       labelled "lap" and not Gamma: this path sum includes the junction steps, while the paper's Gamma (gamma_bars in
       results/summary_heads.json, median 0.96 T) sums the bars only.
The canvas is 7.0 x 2.135 in, printed at \\textwidth as the title-block teaser on page 1 of paper/main.tex.
Layout ANG = 60 deg, chosen with results/probe_dav2-s.csv: DA-S reads all three bars of control 1 with the objects
(bar changes / its jump at control 1's gap -0.27, -0.28, -0.14), so the bend on the impossible image (-0.17, +0.13,
+0.34) is not a bar it also misreads on the possible object; its excess tears are typical (+0.02, +0.05, -0.05); and
the head's lap (0.94 T) equals its median over the 288 impossible images (0.94 T, results/summary_heads.json
gamma_path). The other layouts with head outputs: at 0 deg DA-S reads bar 1 of control 1 nearly flat (-0.02, in units
of its mean jump at the three controls' gaps) and the lap is 0.93 T; at 30 deg DA-S reads bar 3 of control 1 reversed
(+0.65; lap 0.87 T); at 90 deg its e_1 = +0.25 and one bar of the head reads at about half its slope (lap 0.61 T).
(Per-layout values: results/probe_dav2-s.csv, read as in controls() below.)
DA-S maps: outputs/probe/dav2-s/<stim>__none.npy when present (layout 90 only), else DA-S is run here (CPU, a few
seconds; weights from the local Hugging Face cache, downloaded on first use); its cube profiles match
results/probe_dav2-s.csv (checked below).
Needs the stimuli A_a060_p0_t0 and B_a060_k{0,1,2} in data/stim2 (scripts/make_stimuli.py; set PDEPTH_DATA to the
folder holding data/ if it is not the repo).
Run from the repo root: python figures/fig_hero.py -> figures/out/fig_hero.{pdf,png}"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import scipy.sparse as sp
from scipy.ndimage import rotate

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import pdepth.oneform as of
from pdepth.readout import cube_masks, cube_profile
from pdepth.loop import bars_and_joints
from pdepth.render import render_side_trimmed, V, E1, E2
from style import MODELS, setup

setup()
ROOT = os.environ.get("PDEPTH_DATA", ".")
OUT, RES = os.path.join(HERE, "out"), os.path.join(REPO, "results")
ANG, OBJ = 60, 0                                    # layout; object k = 0 (object 1, gap at J1)
IMP = f"A_a{ANG:03d}_p0_t0"
NET, HEAD = "dav2-s", "1form-s-seed0"
NET_REPO = "depth-anything/Depth-Anything-V2-Small-hf"   # NET's weights (pdepth.models.REGISTRY)
ONE, EXACT = "#01665e", "k"                         # 1-form colour as in figures/fig_main.py
NETC, CTLC = "#3182bd", "0.62"                      # DA-S on the impossible image / on control OBJ + 1
T = of.T
_model = []


def side_view(k):
    """side view of object k, 12 deg off the special view, cached for layout 90 (rendered by ray marching when the
    cache is missing), turned to layout ANG"""
    f = os.path.join(OUT, f"side_trimmed_{k}.png")
    if not os.path.exists(f):
        a = np.deg2rad(12)
        img = render_side_trimmed(6, k, view=np.cos(a) * V + np.sin(a) * E1, up=E2, size=360, steps=600)
        os.makedirs(OUT, exist_ok=True)
        plt.imsave(f, np.rot90(img, k=1))                       # layout 90
    img = plt.imread(f)
    d = ANG - 90
    if d % 90 == 0:
        return np.rot90(img, k=d // 90)
    return np.clip(rotate(img, d, reshape=False, order=1, mode="constant", cval=1.0), 0, 1)


def hf_cached(repo_id):
    """are the weights of repo_id in the local Hugging Face cache? (checked on disk, before huggingface_hub is
    imported, so HF_HUB_OFFLINE can still be set)"""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
    snap = os.path.join(hub, "models--" + repo_id.replace("/", "--"), "snapshots")
    return os.path.isdir(snap) and any(os.scandir(snap))


def net_depth(name):
    f = f"{ROOT}/outputs/probe/{NET}/{name}__none.npy"
    if os.path.exists(f):
        return np.load(f).astype(float)
    if not _model:
        if hf_cached(NET_REPO):                         # cached: no network access; otherwise downloaded once
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from pdepth.models import REGISTRY
        _model.append(REGISTRY[NET]())
    return _model[0](np.load(f"{ROOT}/data/stim2/{name}.npz")["img"]).astype(float)


def controls():
    """the network on the three possible controls of this layout: the jump at each junction j on the control with its
    gap at j (tear) and the mean jump there on the other two (ordinary step), as in scripts/analyze_excess.py"""
    J, prof = [], []
    for k in range(3):
        s = dict(np.load(f"{ROOT}/data/stim2/B_a{ANG:03d}_k{k}.npz"))
        prof.append(cube_profile(net_depth(f"B_a{ANG:03d}_k{k}"), cube_masks(s["cube"], 15)))
        J.append(bars_and_joints(prof[-1])[1])
    tear = np.array([J[j][j] for j in range(3)])
    real = np.array([np.mean([J[k][j] for k in range(3) if k != j]) for j in range(3)])
    return tear, real, prof


def check_csv(name, prof):
    """the cube profile computed here against the one in results/probe_<NET>.csv (intact condition)"""
    import csv
    for r in csv.DictReader(open(os.path.join(RES, f"probe_{NET}.csv"))):
        if r["stim"] == name and r["cond"] == "none":
            ref = np.array([float(r[f"prof{i}"]) for i in range(15)])
            return float(np.abs(ref - prof).max() / np.abs(ref).max())
    return float("nan")


def walk(G, om, path):
    """running sum of the edge values along a node path (an edge walked q -> p counts -om), starting at 0"""
    n = G["n"]
    O = sp.csr_matrix((np.r_[om, -om], (np.r_[G["p"], G["q"]], np.r_[G["q"], G["p"]])), shape=(n, n))
    return np.r_[0.0, np.cumsum(np.asarray(O[path[:-1], path[1:]]).ravel())]


def walk_x(lab):
    """position along the loop of every path node (cube c spans [c - 0.5, c + 0.5]); the path starts and ends at the
    central node of cube 0, at x = 0 and x = 15"""
    runs = np.split(np.arange(len(lab)), np.nonzero(np.diff(lab))[0] + 1)
    assert [int(lab[r[0]]) for r in runs] == list(range(15)) + [0], "path does not visit the cubes in loop order"
    x = np.empty(len(lab))
    for i, r in enumerate(runs):
        u = np.arange(len(r))
        x[r] = (0.5 * u / len(r) if i == 0 else 14.5 + 0.5 * (u + 1) / len(r) if i == 15
                else i - 0.5 + (u + 0.5) / len(r))
    return x


def bar_line(prof, j, x):
    """bar j's line (fitted to its four interior cubes, as in pdepth.loop) at local position x in [0, 5]"""
    return np.polyval(np.polyfit(np.arange(1, 5), prof[j * 5 + 1:(j + 1) * 5], 1), x)


def draw_profile(ax, prof, color, label=None, lw=1.3, z=3, ms=2.6):
    """per-cube values (corner cubes hollow: not used by the read-out) and each bar's fitted line; the profile is
    shifted so that bar 1's line starts at 0, so one lap ends where bar 3's line ends (x = 15) and the jump back at J1
    is the distance from there to 0"""
    x = np.arange(15)
    corner = x % 5 == 0
    ax.plot(x[~corner], prof[~corner], "o", color=color, ms=ms, ls="none", zorder=z)
    ax.plot(x[corner], prof[corner], "o", color=color, mfc="white", ms=ms, mew=0.7, ls="none", zorder=z)
    for j in range(3):
        ax.plot(j * 5 + np.array([0, 5]), bar_line(prof, j, np.array([0, 5])), "-", color=color, lw=lw,
                zorder=z - 1, label=label if j == 0 else None)


def vgap(ax, x, y0, y1, text, color, dx=0.25, ha="left"):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="<->", color=color, lw=1.0,
                                                                 shrinkA=0, shrinkB=0))
    ax.text(x + dx, 0.5 * (y0 + y1), text, color=color, fontsize=7.5, va="center", ha=ha)


if __name__ == "__main__":
    st = dict(np.load(f"{ROOT}/data/stim2/{IMP}.npz"))
    masks = cube_masks(st["cube"], 15)
    px = of.pixel_size(st)

    # (i) exact object, (ii) network on the impossible image and on control OBJ + 1, in units of T
    gt = cube_profile(st["depth"][OBJ], masks)
    gt = (gt - bar_line(gt, 0, 0)) / T
    dn = net_depth(IMP)
    tear, real, cprof = controls()
    Tn = float(tear[OBJ])                     # unit: DA-S's jump at the gap of control OBJ + 1 (its "tear" arrow is 1)
    pn = cube_profile(dn, masks)
    dev = check_csv(IMP, pn)
    e = (bars_and_joints(pn)[1] - real) / (tear - real)
    pn = (pn - bar_line(pn, 0, 0)) / Tn
    pc = (cprof[OBJ] - bar_line(cprof[OBJ], 0, 0)) / Tn

    # (iii) the head's depth changes summed once around the loop, and the exact ones on the same path (its output
    # outputs/heads_out/<HEAD>/<IMP>.npz, kept on the figure's pixels in results/; only those are read)
    hf = os.path.join(RES, f"fig1_head-{HEAD}_{IMP}.npz")
    if not os.path.exists(hf):
        r = np.load(f"{ROOT}/outputs/heads_out/{HEAD}/{IMP}.npz")
        np.savez_compressed(hf, kind=r["kind"], g=np.where(st["fg"][..., None], r["g"], 0),
                            edge=np.where(st["fg"], r["edge"], 0))
    h = np.load(hf)
    gh = h["g"].astype(float)
    G = of.graph(st["fg"] & (h["edge"] < 0.5))
    lab = st["cube"][G["r"], G["c"]]
    om = of.edge_form(G, gh, px)
    path = of.loop_path(G, lab, 15, np.isfinite(om))
    phi = walk(G, np.nan_to_num(om), path) / T
    phie = walk(G, np.nan_to_num(of.edge_form(G, of.exact_g(st, px), px)), path) / T
    xw = walk_x(lab[path])
    lap = -phi[-1]
    # the same sum started at J2 (the first path node of corner cube 5): the walk wraps through J1 without a seam, and
    # the seam of one lap moves to J2
    i2 = int(np.nonzero(lab[path] == 5)[0][0])
    idx = np.arange(len(path))
    phi2 = np.where(idx >= i2, phi - phi[i2], phi + (phi[-1] - phi[i2]))
    print(f"{IMP}: head lap {lap:.3f} T (exact changes {-phie[-1]:.3f} T) | {NET}: e {np.round(e, 2)}, own T {Tn:.2f} "
          f"(jumps at real gaps {np.round(tear, 2)}, ordinary steps {np.round(real, 2)}), bar changes / own T "
          f"impossible {np.round(5 * bars_and_joints(pn)[0], 2)}, control {OBJ + 1} "
          f"{np.round(5 * bars_and_joints(pc)[0], 2)} | profile vs results/probe_{NET}.csv: max rel. dev. {dev:.1e}")

    plt.rcParams.update({"axes.titlesize": 7.5})
    fig = plt.figure(figsize=(7.0, 2.135))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.0], hspace=0.5, left=0.072, right=0.995, top=0.883, bottom=0.092)
    gt3 = gs[0].subgridspec(1, 3, wspace=0.08)                    # the top row in the bottom row's three columns, so
    gb = gs[1].subgridspec(1, 3, wspace=0.08)                     # each picture sits above the panel that plots it
    g_obj = gt3[0, 0].subgridspec(1, 3, wspace=0.04)              # over (i): the three objects
    g_net = gt3[0, 1].subgridspec(1, 2, wspace=0.06)              # over (ii): the image and DA-S's depth on it
    g_head = gt3[0, 2].subgridspec(1, 2, wspace=0.06)             # over (iii): the head's sum from J1 and from J2

    # ---- top row
    for k in range(3):
        ax = fig.add_subplot(g_obj[0, k]); ax.imshow(side_view(k)); ax.axis("off")
        ax.set_title(f"object {k + 1}\n(gap at J{k + 1})")
    ax = fig.add_subplot(g_net[0, 0]); ax.imshow(st["img"]); ax.axis("off"); ax.set_title("one image\n(special view)")
    for j, (cx, cy) in enumerate(st["corner_xy"]):                # junction j + 1 sits at corner cube 5j
        p = np.clip(255.5 + 230 * np.array([cx, -cy]) / np.hypot(cx, cy), 40, 471)   # label kept inside the image
        ax.text(p[0], p[1], f"J{j + 1}", color="white", fontsize=7, ha="center", va="center", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.4, foreground="0.2")])
    rr, cc = np.nonzero(st["fg"])                                 # the maps: cropped to the figure, square
    half = max(np.ptp(rr), np.ptp(cc)) / 2 + 12
    lim = lambda ax: (ax.set_xlim(cc.mean() - half, cc.mean() + half), ax.set_ylim(rr.mean() + half, rr.mean() - half))
    ax = fig.add_subplot(g_net[0, 1])
    v = np.where(st["fg"], dn, np.nan); lo, hi = np.nanpercentile(v, [2, 98])
    ax.imshow(v, cmap="turbo_r", vmin=lo, vmax=hi); ax.axis("off"); lim(ax)
    ax.set_title("DA-S depth\n(red: near)")
    # the head's depth changes summed along the walk from J1 (phi, in units of T) and from J2 (phi2), spread to every
    # pixel of the figure from the nearest path node of the same cube; the walk's first cube holds both of its ends, so
    # the sum meets itself there with a seam of one lap; same colour map as the DA-S map (red: near)
    from scipy.spatial import cKDTree
    pr, pc_ = G["r"][path].astype(float), G["c"][path].astype(float)
    r, c = pr, pc_
    runs = np.split(np.arange(len(path)), np.nonzero(np.diff(lab[path]))[0] + 1)
    for col, (vals, start) in enumerate([(phi, 1), (phi2, 2)]):
        ax = fig.add_subplot(g_head[0, col])
        acc = np.full(st["fg"].shape, np.nan)
        for q in range(15):
            nq = np.nonzero(lab[path] == q)[0]
            m = st["fg"] & (st["cube"] == q)
            if len(nq) and m.any():
                rr_, cc_ = np.nonzero(m)
                acc[rr_, cc_] = vals[nq[cKDTree(np.c_[pr[nq], pc_[nq]]).query(np.c_[rr_, cc_])[1]]]
        ax.imshow(acc, cmap="turbo_r", vmin=-lap - 0.03, vmax=0.03)
        # the path, drawn through the mean position of its nodes in each cube (it is a 4-neighbour staircase), and
        # the direction of travel
        ax.plot([c[0]] + [c[q].mean() for q in runs[1:-1]] + [c[0]],
                [r[0]] + [r[q].mean() for q in runs[1:-1]] + [r[0]], color="k", lw=0.6)
        for j in range(3):
            a, b = [(np.mean(c[lab[path] == q]), np.mean(r[lab[path] == q])) for q in (5 * j + 2, 5 * j + 3)]
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="k", lw=0.6, mutation_scale=7,
                                                             shrinkA=0, shrinkB=0))
        cx, cy = st["corner_xy"][start - 1]                       # the walk's start and seam, labelled
        p = np.array([cc.mean(), rr.mean()]) + (half - 16) * np.array([cx, -cy]) / np.hypot(cx, cy)
        ax.text(p[0], p[1], f"J{start}", color="white", fontsize=7, ha="center", va="center", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.4, foreground="0.2")])
        ax.axis("off"); lim(ax)
        ax.set_title("depth-change head,\nsummed from J1" if start == 1 else "the same sum,\nfrom J2")

    # ---- bottom row: the loop walked once
    axs = [fig.add_subplot(gb[0, i]) for i in range(3)]
    for a in axs[1:]:
        a.sharey(axs[0])
    a = axs[0]
    draw_profile(a, gt, EXACT, None)
    jb = (OBJ - 1) % 3                                            # the bar that ends at the gap, and the next one
    vgap(a, 15.0 if OBJ == 0 else 5 * OBJ, bar_line(gt, jb, 5), bar_line(gt, OBJ, 0), "$+T$", EXACT,
         dx=-0.35 if OBJ == 0 else 0.35, ha="right" if OBJ == 0 else "left")
    a.set_title(f"(i) object {OBJ + 1}, exact depth\nper cube (gap at J{OBJ + 1})")
    a = axs[1]
    draw_profile(a, pc, CTLC, f"control {OBJ + 1}", lw=0.9, z=2, ms=2.2)
    draw_profile(a, pn, NETC, "impossible image")
    if OBJ == 0:                                                  # the control's tear at J1, the end of the walk
        vgap(a, 15.0, bar_line(pc, 2, 5), 0.0, "tear", CTLC, dx=-0.3, ha="right")
    fmt = lambda x: f"{round(float(x), 2) + 0.0:+.2f}".replace("-", chr(0x2212))
    for j, x in enumerate((0, 5, 10)):
        a.text(x + 0.15, 0.97, f"$e_{j + 1}$ {fmt(e[j])}", transform=a.get_xaxis_transform(), fontsize=7,
               color="0.25", va="top")
    a.legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.4, borderaxespad=0.1)
    a.set_title("(ii) DA-S depth per cube, in units\nof its jump at a real gap")
    a = axs[2]
    a.plot(xw, phie, "--", color="0.55", lw=0.9, label="exact")
    a.plot(xw, phi, "-", color=ONE, lw=1.4, label="depth-change head")
    vgap(a, 15.0, 0.0, -lap, f"lap: ${lap:.2f}\\,T$", ONE, dx=-0.35, ha="right")
    a.set_title("(iii) depth-change head, summed\nalong the walk from J1")
    a.legend(frameon=False, fontsize=7, loc="lower left", handlelength=1.4, borderaxespad=0.1)
    for a in axs:
        a.axhline(0, color="0.6", lw=0.6, ls=":", zorder=0)
        for x in (0, 5, 10, 15):                                  # junction j + 1 at corner cube 5j (and J1 at 15)
            a.axvline(x, color="0.88", lw=0.6, zorder=0)
        a.set_xlim(-0.7, 15.6)
        a.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
        a.set_xticklabels(["J1", "bar 1", "J2", "bar 2", "J3", "bar 3", "J1"], fontsize=7)
        a.tick_params(axis="x", length=0, pad=2)
    for a in axs[1:]:
        a.tick_params(axis="y", labelleft=False)
    lo_y = min(gt.min(), pn.min(), pc.min(), phi.min(), phie.min(), bar_line(pc, 2, 5))
    hi_y = max(gt.max(), pn.max(), pc.max(), phi.max(), phie.max())
    axs[0].set_ylim(lo_y - 0.12, hi_y + 0.45)
    axs[0].set_yticks([-1, -0.5, 0])
    axs[0].set_yticklabels([chr(0x2212) + "1", chr(0x2212) + "0.5", "0"])
    axs[0].set_ylabel("depth / $T$\nnear $\\leftarrow\\ \\rightarrow$ far", fontsize=7.5, labelpad=2)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_hero.pdf"), dpi=300); fig.savefig(os.path.join(OUT, "fig_hero.png"), dpi=220)
