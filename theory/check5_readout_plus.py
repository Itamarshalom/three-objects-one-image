"""Check 5 (docs/theory.md Sec. 6): the span diagnostic (split, ramp share, residual) applied to the pilot predictions.
It separates the theory's cases on noise-free maps, not at the pilot's error level (check9).

Per-cube profile y (15 values) is fitted by least squares with the three object profiles and a constant:
  y ~ sum_k w_k P_k + c.
The span of {P_1, P_2, P_3, 1} is exactly 'every bar a straight ramp with a common slope, any bar offsets'.
  resid      = |y - fit| / |y - mean y|   (0 for any mixture, object or affine image of one; large for bent bars
                                           or steps inside bars, e.g. the SSI-MAE patchwork)
  w / sum w  = tear split of the fitted part (= p for a posterior mean; one-hot for a committed object)
  ramp share = sum w / sum |w|           (1 for any convex mixture; < 1 when some weights are negative, i.e.
                                           the bars are flatter than the tears require, as for the SSI-L2
                                           principal component; DAv2's one-answer controls show it too, check9)"""
import numpy as np
from common import pilot_profiles, object_profiles

rows = []
for nm, y in pilot_profiles("*texture_a90*.npy"):     # pilot outputs (bundled profiles if the maps are absent)
    P = object_profiles("texture_a90_possible0" if "possible0" in nm else "texture_a90")
    n = len(y)
    A = np.concatenate([P.T, np.ones((n, 1))], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    w = c[:3]
    resid = np.linalg.norm(A @ c - y) / np.linalg.norm(y - y.mean())
    split = w / w.sum()
    ramp = w.sum() / np.abs(w).sum()
    rows.append((nm, split, ramp, resid))
    print(f"{nm:42s} split(w)={np.round(split, 2)}  ramp share={ramp:+.2f}  profile resid={resid:.2f}")
