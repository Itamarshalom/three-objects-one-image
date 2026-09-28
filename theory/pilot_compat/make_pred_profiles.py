"""Bundle what check5, check9 and check10 need from the pilot network outputs (not in the repository, ~100 MB):
the 15-cube profile of every pilot output on texture_a90 and its possible control (pilot read-out, tears.py), and
the object's share of the frame's 2/98 depth range for the Marigold outputs, and the three objects' profiles on the
possible control's stimulus (not included either).
usage: python make_pred_profiles.py <pilot folder with pred/ and stim/>   -> pred_profiles.json next to this file"""
import os, sys, glob, json
import numpy as np
from tears import cube_profile

HERE = os.path.dirname(os.path.abspath(__file__))
pilot = sys.argv[1]
stims = {nm: dict(np.load(os.path.join(pilot, "stim", f"{nm}.npz"))) for nm in ("texture_a90", "texture_a90_possible0")}
fg = stims["texture_a90"]["fg"]
out = {"profile": {}, "share98": {}, "objects": {}}
for nm, st in stims.items():                     # the three objects' profiles on each stimulus (cube labels differ)
    out["objects"][nm] = [cube_profile(st["depth"][k], st["cube"], int(st["n_cubes"])).tolist() for k in range(3)]
for f in sorted(glob.glob(os.path.join(pilot, "pred", "*texture_a90*.npy"))):
    nm = os.path.basename(f)[:-4]
    st = stims["texture_a90_possible0" if "possible0" in nm else "texture_a90"]
    d = np.load(f)
    out["profile"][nm] = cube_profile(d, st["cube"], int(st["n_cubes"])).tolist()
    if nm.startswith("mg10"):
        a, b = np.percentile(d, [2, 98]); lo, hi = np.percentile(d[fg], [2, 98])
        out["share98"][nm] = float((hi - lo) / (b - a))
json.dump(out, open(os.path.join(HERE, "pred_profiles.json"), "w"), indent=0)
print(f"{len(out['profile'])} profiles, {len(out['share98'])} Marigold range shares")
