"""Attention cut: the erasure probe (scripts/run_probe.py) done inside a Depth Anything V2 network instead of on the
pixels. Every impossible image and possible control is run as it is and with junction j cut out of attention in every
encoder block; the pixels, the embeddings and the DPT decoder are untouched.

Junction j's tokens are the patches of the 37x37 token grid that are more than `cover` (default 0.25, the share of
their pixels) inside the region the pixel erasure removes (pdepth/probe.py: cubes 5j-1, 5j, 5j+1 dilated by 6 px).
  keys    : no token (CLS included) may attend to a junction token; junction tokens still attend to all other tokens
            and to themselves, and the decoder still reads them: only the flow of information out of them is cut
  confine : as keys, and the junction tokens attend only to each other (no information in or out)
  bars    : only the tokens of the two bars meeting at j (more than `cover` on bar j-1's interior cubes 5j-4..5j-1,
            or on bar j's 5j+1..5j+4; corner cubes belong to neither) may not attend to each other, both ways
The cut is exact: the masked attention weights are set to -inf before the softmax and asserted to be exactly zero in
every block; on the first image the eager attention used for the cut is checked against the stock attention and
(keys, confine) the other tokens' last hidden states are checked to be bitwise independent of the junction tokens.
usage: python scripts/run_attention_cut.py <dav2-s|dav2-b|dav2-l> [stim_glob] [keys|confine|bars] [cover]  (THREADS=n)
writes results/attn<-mode><ATTN_TAG>_<model>.csv (no -mode for keys; ATTN_TAG from the environment, e.g. _loops) in
the layout of results/probe_<model>.csv (cond none / a0 a1 a2, one row per image and condition): prof0..14 on the
full cube masks, hprof0..14 with the pixel probe's hole removed (the cubes the pixel erasure leaves visible); n_tok =
junction tokens (keys, confine) or bar tokens (bars). Maps of layout 90 (FIG below) go to
outputs/attn<-mode>/<model>/<stim>__<cond>.npy (float16). Summary: scripts/analyze_attention.py
"""
import os, sys, glob, csv, time, types, numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdepth.models import REGISTRY
from pdepth.readout import cube_masks, cube_profile
from pdepth.probe import erase, junction_cubes, N_BAR
from run_features import IDX, G

META = ["set", "angle", "perm", "tex", "mirror", "possible", "tau", "nuis", "style"]
FIG = {"A_a090_p0_t0", "B_a090_k0", "B_a090_k1", "B_a090_k2"}
CUT = {"mask": None, "checked": 0}                     # mask [N, N] bool: True = query (row) may not read key (column)


def cut_attention(self, hidden_states, **kw):
    """Dinov2SelfAttention.forward; eager attention with the masked weights set to exactly zero when a cut is active"""
    mask = CUT["mask"]
    if mask is None:
        return type(self).forward(self, hidden_states, **kw)
    sh = hidden_states.shape[0], -1, self.num_attention_heads, self.attention_head_size
    q, k, v = (f(hidden_states).view(*sh).transpose(1, 2) for f in (self.query, self.key, self.value))
    w = torch.matmul(q, k.transpose(2, 3)) * self.scaling
    w = w.masked_fill(mask, float("-inf")).softmax(-1)
    assert not w[:, :, mask].any(), "a masked attention weight is not zero"
    CUT["checked"] += 1
    o = torch.matmul(w, v).transpose(1, 2).contiguous()
    return o.reshape(o.shape[0], o.shape[1], -1), w


def patch_share(region):
    """share of each patch's pixels inside a 512 px region, on the token grid [G, G]"""
    return region[IDX][:, IDX].reshape(G, 14, G, 14).mean((1, 3))


def cut_mask(st, j, mode, cover):
    """[N, N] attention mask of junction j (N = 1 + G*G, token 0 = CLS) and the tokens it cuts (keys, confine: the
    junction tokens; bars: the tokens of both bars)"""
    n_cubes = int(st["n_cubes"])
    tok = lambda share: 1 + np.where((share > cover).ravel())[0]
    mask = torch.zeros(1 + G * G, 1 + G * G, dtype=torch.bool)
    if mode in ("keys", "confine"):
        t = torch.as_tensor(tok(patch_share(erase(st, junction_cubes(j, n_cubes))[1])), dtype=torch.long)
        mask[:, t] = True                               # nobody reads junction tokens
        if mode == "confine":
            mask[t] = True; mask[t[:, None], t[None, :]] = False      # junction tokens read only each other
        mask[t, t] = False                              # every junction token reads itself
        return mask, t
    bar = lambda cubes: tok(patch_share(np.isin(st["cube"], [c % n_cubes for c in cubes])))
    s1 = bar(range(N_BAR * j - N_BAR + 1, N_BAR * j)); s2 = bar(range(N_BAR * j + 1, N_BAR * j + N_BAR))
    both = np.intersect1d(s1, s2)                       # a patch on both bars (none in the stimuli) is left alone
    s1, s2 = (torch.as_tensor(np.setdiff1d(s, both), dtype=torch.long) for s in (s1, s2))
    mask[s1[:, None], s2[None, :]] = True
    mask[s2[:, None], s1[None, :]] = True
    return mask, torch.cat([s1, s2])


@torch.no_grad()
def self_check(m, st, mode, cover):
    """eager cut attention with an empty mask vs the stock attention; modes keys, confine: the other tokens' last hidden
    states must not depend on what the junction tokens hold (exact zeros in the attention make this bitwise)"""
    ref = m(st["img"])
    CUT["mask"] = torch.zeros(1 + G * G, 1 + G * G, dtype=torch.bool, device=m.dev)
    eager = m(st["img"])
    print(f"eager vs stock attention, no cut: max |diff| {np.abs(eager - ref).max():.2e} (depth range "
          f"{np.ptp(ref):.2f})", flush=True)
    mask, t = cut_mask(st, 0, mode, cover)
    CUT["mask"], t = mask.to(m.dev), t.to(m.dev)
    if mode != "bars":
        bb = m.model.backbone
        pix = m.proc(images=Image.fromarray(st["img"]), return_tensors="pt")["pixel_values"].to(m.dev)
        outs = []
        for noise in (0.0, 1.0):
            h = bb.embeddings(pix)
            h[:, t] += noise * h.std() * torch.randn(h[:, t].shape, generator=torch.Generator().manual_seed(0)).to(m.dev)
            for lyr in bb.encoder.layer:
                h = lyr(h)
            outs.append(h)
        other = torch.ones(1 + G * G, dtype=torch.bool, device=m.dev); other[t] = False
        d_other = float((outs[0][:, other] - outs[1][:, other]).abs().max())
        d_junc = float((outs[0][:, t] - outs[1][:, t]).abs().max())
        print(f"junction 0 ({len(t)} tokens) cut, noise added to the junction tokens' embeddings: other tokens' "
              f"block-{len(bb.encoder.layer)} states max |diff| {d_other:.1e} (junction tokens: {d_junc:.2f})", flush=True)
        assert d_other == 0, "information leaks out of the cut junction"
    CUT["mask"] = None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    key = sys.argv[1]
    stims = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else "data/stim2/[AB]_*.npz"))
    mode = sys.argv[3] if len(sys.argv) > 3 else "keys"
    cover = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    assert mode in ("keys", "confine", "bars"), mode
    if not stims:
        sys.exit("no stimuli found: run scripts/make_stimuli.py first")
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    m = REGISTRY[key]()
    if type(m.model.backbone).__name__ != "Dinov2Backbone":
        sys.exit(f"{key}: the attention cut is written for the DINOv2 backbone of Depth Anything V2")
    for lyr in m.model.backbone.encoder.layer:
        a = lyr.attention.attention
        a.forward = types.MethodType(cut_attention, a)
    tag = ("" if mode == "keys" else "-" + mode) + os.environ.get("ATTN_TAG", "")
    out_dir = f"outputs/attn{tag}/{key}"
    os.makedirs(out_dir, exist_ok=True); os.makedirs("results", exist_ok=True)
    rows, t0 = [], time.time()
    for i, f in enumerate(stims):
        name = os.path.basename(f)[:-4]
        st = dict(np.load(f))
        n_cubes, n_bars = int(st["n_cubes"]), int(st["n_bars"]) if "n_bars" in st else 3
        if i == 0:
            self_check(m, st, mode, cover)
        masks = cube_masks(st["cube"], n_cubes)
        for cond in ["none"] + [f"a{j}" for j in range(n_bars)]:
            n_tok, hole = 0, None
            if cond != "none":
                j = int(cond[1:])
                mask, tok = cut_mask(st, j, mode, cover)
                CUT["mask"], n_tok = mask.to(m.dev), len(tok)
                hole = erase(st, junction_cubes(j, n_cubes))[1]
            depth = m(st["img"])
            CUT["mask"] = None
            if name in FIG:
                np.save(f"{out_dir}/{name}__{cond}.npy", depth.astype(np.float16))
            prof = cube_profile(depth, masks)
            hprof = prof if hole is None else cube_profile(depth, [mk & ~hole for mk in masks])
            rows.append(dict(model=key, stim=name, cond=cond, seed=0, **{k: (st[k].item() if k in st else "") for k in META},
                             n_bars=n_bars, mode=mode, cover=cover, n_tok=n_tok,
                             **{f"prof{j}": v for j, v in enumerate(prof)}, **{f"hprof{j}": v for j, v in enumerate(hprof)}))
        if (i + 1) % 25 == 0:
            print(f"{key} {mode}: {i + 1}/{len(stims)} ({time.time() - t0:.0f}s)", flush=True)
    with open(f"results/attn{tag}_{key}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
        w.writeheader(); w.writerows(rows)
    print(f"done {key} {mode}: {len(rows)} rows, {CUT['checked']} masked attention maps checked to be exactly zero "
          f"({time.time() - t0:.0f}s)")
