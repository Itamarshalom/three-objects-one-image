"""print a compact table of every population-optimisation result in opt/*.json (check3b) and opt7/*.json (check7)"""
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(os.path.join(HERE, "opt", "*.json"))) + sorted(glob.glob(os.path.join(HERE, "opt7", "*.json"))):
    r = json.load(open(f))
    c = r["candidates"]
    objs = {k: v for k, v in c.items() if k.startswith("obj")}
    kbest = min(r["runs"], key=lambda k: r["runs"][k]["loss"])
    b = r["runs"][kbest]
    hb = r.get("halfbar_top5", [])
    nonobj = [x for x in hb if len(set(x[1])) > 1] or ([r["halfbar_best_nonobject"]] if "halfbar_best_nonobject" in r else [])
    print(f"{r['loss']:6s} p={[round(x, 3) for x in r['p']]} | objects " + " ".join(f"{k}={v:.4f}" for k, v in objs.items()) +
          f" | mean={c.get('mean', float('nan')):.4f} middle={c.get('middle', float('nan')):.4f}" +
          (f" | best non-object half-bar {nonobj[0][0]:.4f} {tuple(nonobj[0][1])}" if nonobj else "") +
          f" | best after Adam: from {kbest} loss={b['loss']:.4f} split={b['split']} resid={b['span_resid']}" +
          " | Adam ends: " + ", ".join(f"{k}:{v['loss']:.4f}" for k, v in r["runs"].items()))
