"""Check 13 (docs/theory.md Sec. 4c): are the leading-spacing numbers of failure mode 4
(1 step: q = (0.33, 0.34, 0.33) whatever p; 4 steps: (0.665, 0.279, 0.056)) geometry-independent?  check4b code.
Geometries: two idealised ones and the encoded ones of vae_geometry_flat.json (written by check10)."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "check4b_ddim.py"), encoding="utf-8").read().split("PS = [")[0]
ns = {}
exec(src, ns)
run = ns["run"]
vg = json.load(open(os.path.join(HERE, "vae_geometry_flat.json")))
geo = {"ideal |dz|=1": np.full((3, 3), 1e4) + np.eye(3) * 0.5, "ideal |dz|=40": np.full((3, 3), 1e4) + np.eye(3) * 800}
geo.update({f"encoded {k}": np.array(v["gram"]) for k, v in vg.items()})
for nm, G in geo.items():
    out = []
    for p in [(0.6, 0.3, 0.1), (0.1, 0.3, 0.6)]:
        out += [f"p={p} S={S}: q={np.round(run(G, np.array(p), S, spacing='leading')['q'], 3)}" for S in [1, 4]]
    print(f"{nm:18s} " + " | ".join(out), flush=True)
