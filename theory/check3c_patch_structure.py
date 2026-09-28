"""Check 3c (docs/theory.md Sec. 3.2): where the SSI-MAE optimum (all p_k < 1/2) takes which object's value, per cube along the loop."""
import numpy as np
from common import load, mad_norm, cube_profile, to_img, ensure
TAGS = ["0.40_0.35_0.25", "0.36_0.33_0.31", "0.45_0.45_0.10", "0.33_0.33_0.33"]
ensure([f"patch_{t}.npy" for t in TAGS], ["check3_l1.py"])         # patchworks written by check3_l1.py
st = load("texture_a90"); fg = st["fg"]; cube = st["cube"][fg]; n = st["n"]
X = st["depth"][:, fg]; Nn = np.array([mad_norm(X[k]) for k in range(3)])
mid = np.sort(Nn, 0)[1]
for tag in TAGS:
    g = np.load(f"patch_{tag}.npy")   # built from the normalised object values (not re-normalised)
    lab = np.full(g.shape, "?", dtype=object)
    for k in range(3):
        lab[np.abs(g - Nn[k]) < 1e-6] = str(k + 1)
    ismid = np.abs(g - mid) < 1e-6
    print(f"p={tag}: per cube (0..14, corner cubes 0,5,10) share of pixels at obj1/obj2/obj3 value, 'm' = share at the middle map")
    rows = []
    for c in range(n):
        s = cube == c
        sh = [np.mean(np.abs(g[s] - Nn[k][s]) < 1e-6) for k in range(3)]
        rows.append(f"c{c:02d}:" + "/".join(f"{x:.2f}" for x in sh) + f" m{np.mean(ismid[s]):.2f}")
    for i in range(0, n, 5):
        print("   ", "  ".join(rows[i:i + 5]))
    P = cube_profile(to_img(g, fg), st["cube"], n)
    print("    per-cube profile:", np.round(P, 2))
    print("    per-cube steps  :", np.round(np.roll(P, -1) - P, 2))
