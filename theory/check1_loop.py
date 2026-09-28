"""Check 1: loop structure of the three valid depth maps and linearity of the tear read-out."""
import numpy as np
from common import load, split_of, excess_of, cube_profile, to_img, mad_norm

st = load("texture_a90")
fg, cube, n, corners = st["fg"], st["cube"], st["n"], st["corners"]
D = st["depth"]                                   # [3, H, W], larger = farther
nm1 = (n + 3) // 3                                # cubes per bar side = n_side - 1 ... (15 cubes -> 5 per bar)
n_side = n // 3 + 1                               # the renderer's n (6)
T = (n_side - 1) * np.sqrt(3)
print(f"N cubes = {n}, corners at {corners}, renderer n = {n_side}, T = (n-1)*sqrt(3) = {T:.6f}, "
      f"delta = 1/sqrt3 = {1/np.sqrt(3):.6f}, N*delta = {n/np.sqrt(3):.6f}")

# (a) pixelwise: d_k - d_0 = -T * 1[cube < c_k]
for k in range(3):
    pred = -T * (cube[fg] < corners[k])
    err = np.abs((D[k][fg] - D[0][fg]) - pred).max()
    print(f"obj{k+1}: max |(d_k - d_1) - (-T*1[cube < c_k])| over fg = {err:.2e}")

# (b) per-cube profile, steps, excess matrix E[k, j] = excess of object k at corner j
P = np.array([cube_profile(D[k], cube, n) for k in range(3)])
steps0 = np.roll(P[0], -1) - P[0]
print("per-cube steps of obj1:", np.round(steps0, 4))
E = np.array([excess_of(D[k], st) for k in range(3)])
print("E / T =\n", np.round(E / T, 5))
print("row sums / T:", np.round(E.sum(1) / T, 5))

# (c) mixtures: split(sum p_k d_k) = p^T E / (p^T E 1) ~ p
rng = np.random.default_rng(0)
errs, errs_pred = [], []
for _ in range(200):
    p = rng.dirichlet(np.ones(3))
    m = np.tensordot(p, D, 1)
    s = split_of(m, st)
    errs.append(np.abs(s - p).max())
    errs_pred.append(np.abs(s - (p @ E) / (p @ E).sum()).max())
print(f"200 random mixtures: max|split - p| = {max(errs):.2e}; max|split - p^T E/(p^T E 1)| = {max(errs_pred):.2e}")
# general linear combinations (sum w != 1) and affine invariance
w = np.array([1.3, -0.4, 0.5])
m = np.tensordot(w, D, 1)
print("w =", w, " split =", np.round(split_of(m, st), 5), " w/sum w =", np.round(w / w.sum(), 5),
      " split(-3*m+7) =", np.round(split_of(-3 * m + 7, st), 5))

# (d) Gram of the centred fg maps: within-cube part (alpha) + profile part
X = D[:, fg]
Xc = X - X.mean(1, keepdims=True)
G = Xc @ Xc.T / fg.sum()
cm = np.array([P[0][c] for c in cube[fg]])        # obj1 per-cube mean at each pixel
rho = X[0] - cm                                   # within-cube residual, identical for all objects
print("rho identical for all objects:",
      max(np.abs((X[k] - np.array([P[k][c] for c in cube[fg]])) - rho).max() for k in range(3)))
alpha = rho @ rho / fg.sum()
print("Gram G/M of centred depth (per pixel):\n", np.round(G, 4))
print(f"within-cube variance alpha = {alpha:.4f}; cos between objects =",
      np.round(G / np.sqrt(np.outer(np.diag(G), np.diag(G))), 4)[np.triu_indices(3, 1)])
# circulant fit G ~ c I + gamma J
c_fit = np.diag(G).mean() - G[np.triu_indices(3, 1)].mean()
g_fit = G[np.triu_indices(3, 1)].mean()
print(f"circulant fit: c = {c_fit:.4f}, gamma = {g_fit:.4f}  (sawtooth model: gamma/c = (alpha - beta/3)/(4 beta/3))")
np.save("gram.npy", G)

# (e) median/MAD gauge: on each bar the three normalised maps are ordered far/mid/near with ~T/3 spacing
Nn = np.array([mad_norm(X[k]) for k in range(3)])
print("median/MAD of raw objects:", np.round([np.median(X[k]) for k in range(3)], 4),
      np.round([np.mean(np.abs(X[k] - np.median(X[k]))) for k in range(3)], 4))
bar = np.minimum(cube[fg] // (n // 3), 2)
for b in range(3):
    sel = bar == b
    order = np.argsort(Nn[:, sel], 0)
    uniq, cnt = np.unique(order.T, axis=0, return_counts=True)
    gaps = np.diff(np.sort(Nn[:, sel], 0), axis=0)
    print(f"bar {b}: orderings (low->high obj idx) {[(tuple(u), c) for u, c in zip(uniq, cnt)]}, "
          f"gaps mean {gaps.mean(1).round(4)} std {gaps.std(1).round(5)}")
mid = np.sort(Nn, 0)[1]
mean = Nn.mean(0)
print(f"middle-of-three map vs equal-weight mean (normalised units): max |diff| = {np.abs(mid - mean).max():.4f}, "
      f"rms = {np.sqrt(((mid - mean) ** 2).mean()):.4f}, rms of mean = {np.sqrt((mean ** 2).mean()):.4f}")
print("split(middle map) =", np.round(split_of(to_img(mid, fg), st), 4),
      " split(mean map) =", np.round(split_of(to_img(mean, fg), st), 4))
# raw renderer gauge: pointwise middle is object 2 (two objects coincide on every bar)
midraw = np.sort(X, 0)[1]
print("raw-gauge middle map == obj2 ?", np.abs(midraw - X[1]).max(), " split:", np.round(split_of(to_img(midraw, fg), st), 4))
