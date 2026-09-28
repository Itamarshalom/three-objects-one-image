"""Figure 1: three objects, one image; what an ideal predictor pays; what a network does.
Top: the three trimmed objects 12 deg off the special view, the shared image, the possible control (gap of object 2
visible), the depth of the objects' average, and a network's depth on both. Bottom left: loop profiles an ideal predictor could return (one object =
one full tear; the average = a third of a tear at every joint). Bottom right: a network on the possible image vs
the impossible image at the same layout: identical steps at the shared joints, no tear at joint 2; each joint's
excess tear e_j is printed (1 = the control's tear, 0 = an ordinary step). Corner cubes (hollow) are not used.
Set PDEPTH_DATA to the folder holding data/stim2 and outputs/pred (default: the repo). The side views are cached
in figures/out/side_trimmed_k.png; delete them to re-render with pdepth.render.render_side_trimmed (a few CPU
minutes each)."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdepth.render import render_side_trimmed, V, E1, E2
from pdepth.readout import cube_masks, cube_profile
from pdepth.loop import bars_and_joints
from style import MODELS, setup

setup()
ROOT = os.environ.get("PDEPTH_DATA", ".")
IMP, POS, ANG = "A_a090_p0_t0", "B_a090_k1", 90
NET = "dav2-b"


def side_view(k):
    f = f"figures/out/side_trimmed_{k}.png"
    if not os.path.exists(f):
        a = np.deg2rad(12)
        img = render_side_trimmed(6, k, view=np.cos(a) * V + np.sin(a) * E1, up=E2, size=360, steps=600)
        plt.imsave(f, np.rot90(img, k=ANG // 90))
    return plt.imread(f)


def show_depth(ax, d, fg, title):
    v = np.where(fg, d, np.nan)
    lo, hi = np.nanpercentile(v, [2, 98])
    ax.imshow(v, cmap="turbo_r", vmin=lo, vmax=hi); ax.set_title(title); ax.axis("off")


def draw_loop(ax, prof, color, label, ls="-"):
    """cube medians as dots and each bar's fitted line (extended to its joints)"""
    corner = np.arange(15) % 5 == 0                       # corner cubes: drawn hollow, not used by the read-out
    ax.plot(np.arange(15)[~corner], prof[~corner], "o", color=color, ms=2.4, ls="none")
    ax.plot(np.arange(15)[corner], prof[corner], "o", color=color, mfc="none", ms=2.4, mew=0.6, ls="none")
    for j in range(3):
        t = np.array([0, 5]); a0 = np.polyfit(np.arange(1, 5), prof[j * 5 + 1:(j + 1) * 5], 1)
        ax.plot(j * 5 + t, np.polyval(a0, t), ls, color=color, lw=1.2, label=label if j == 0 else None)


if __name__ == "__main__":
    plt.rcParams.update({"axes.titlesize": 6.8})
    si, sp = dict(np.load(f"{ROOT}/data/stim2/{IMP}.npz")), dict(np.load(f"{ROOT}/data/stim2/{POS}.npz"))
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(2, 8, height_ratios=[1, 1.15], hspace=0.30, wspace=0.08)
    for k in range(3):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(side_view(k)); ax.axis("off"); ax.set_title(f"object {k + 1}")
    ax = fig.add_subplot(gs[0, 3]); ax.imshow(si["img"]); ax.axis("off"); ax.set_title("impossible\n(all three)")
    ax = fig.add_subplot(gs[0, 4]); ax.imshow(sp["img"]); ax.axis("off"); ax.set_title("possible control\n(object 2)")
    show_depth(fig.add_subplot(gs[0, 5]), si["depth"].mean(0), si["fg"], "average of 3\n(ideal L2)")
    dp = np.load(f"{ROOT}/outputs/pred/{NET}/{POS}.npy").astype(float)
    di = np.load(f"{ROOT}/outputs/pred/{NET}/{IMP}.npy").astype(float)
    lab = MODELS[NET]["label"].replace("Depth Anything V2", "DAv2")
    show_depth(fig.add_subplot(gs[0, 6]), dp, sp["fg"], f"{lab}\ncontrol")
    show_depth(fig.add_subplot(gs[0, 7]), di, si["fg"], f"{lab}\nimpossible")
    mi, mp = cube_masks(si["cube"], 15), cube_masks(sp["cube"], 15)
    ax = fig.add_subplot(gs[1, :4])
    g = np.stack([cube_profile(si["depth"][k], mi) for k in range(3)])
    draw_loop(ax, g[1], "k", "object 2: one full tear, at J2")
    draw_loop(ax, g.mean(0) - 6.0, "#e34a33", "average (uniform belief): a third at every joint", ls="--")
    ax.set_title("what an ideal predictor can return")
    ax2 = fig.add_subplot(gs[1, 4:])
    pp, pi = cube_profile(dp, mp), cube_profile(di, mi)
    draw_loop(ax2, pp, "#9ecae1", "possible control (object 2)")
    draw_loop(ax2, pi, MODELS[NET]["color"], "impossible image")
    ax2.set_title(f"{lab}: same steps at J1, J3; the tear at J2 is gone", pad=11)
    # each joint's excess tear e_j (control -> impossible), printed above the panel: e_j = (J_j - r_j) / (g_j - r_j)
    # with r_j, g_j from the network's three controls at this layout (1 = the control's tear, 0 = an ordinary step)
    Jp, Ji = bars_and_joints(pp)[1], bars_and_joints(pi)[1]
    Jc = []
    for k in range(3):
        sk = dict(np.load(f"{ROOT}/data/stim2/B_a{ANG:03d}_k{k}.npz"))
        dk = np.load(f"{ROOT}/outputs/pred/{NET}/B_a{ANG:03d}_k{k}.npy").astype(float)
        Jc.append(bars_and_joints(cube_profile(dk, cube_masks(sk["cube"], 15)))[1])
    g = np.array([Jc[j][j] for j in range(3)])
    r = np.array([np.mean([Jc[k][j] for k in range(3) if k != j]) for j in range(3)])
    ep, ei = (Jp - r) / (g - r), (Ji - r) / (g - r)
    fmt = lambda x: f"{round(float(x), 2) + 0.0:+.2f}".replace("-", "−")
    for j, x in enumerate((-0.5, 4.5, 9.5)):
        ax2.text(x + 0.25, 1.01, f"$e_{j + 1}$: {fmt(ep[j])}$\\to${fmt(ei[j])}", fontsize=6.3, va="bottom",
                 transform=ax2.get_xaxis_transform(), color="k" if j == 1 else "0.25")
    for a in (ax, ax2):
        for c in (-0.5, 4.5, 9.5, 14.5):
            a.axvline(c, color="0.85", lw=0.6, zorder=0)
        a.set_xticks([-0.5, 2, 4.5, 7, 9.5, 12, 14.5])
        a.set_xticklabels(["J1", "bar 1", "J2", "bar 2", "J3", "bar 3", "J1"], fontsize=6)
        a.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        a.set_yticks([])
    ax.set_ylabel("depth\nnear $\\leftarrow\\ \\rightarrow$ far", fontsize=7)
    os.makedirs("figures/out", exist_ok=True)
    fig.savefig("figures/out/fig_hero.pdf", dpi=300); fig.savefig("figures/out/fig_hero.png", dpi=220)
    print("control J:", np.round(bars_and_joints(pp)[1], 2), " impossible J:", np.round(bars_and_joints(pi)[1], 2))
