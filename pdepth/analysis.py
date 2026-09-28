"""Shared analysis helpers: loading read-out tables, bootstrap intervals."""
import csv
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("", None):
                continue
            try:
                r[k] = float(v) if any(c in v for c in ".eE") or k in ("w1", "w2", "w3") else int(v)
            except ValueError:
                if v in ("True", "False"):
                    r[k] = v == "True"
        r["w"] = np.array([r["w1"], r["w2"], r["w3"]], float)
    return rows


def boot_ci(x, f=np.mean, n=2000, seed=0, groups=None):
    """percentile bootstrap CI of f(x); with groups, resample whole groups (cluster bootstrap)"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    if groups is None:
        stats = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    else:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        idx = {g: np.where(groups == g)[0] for g in ug}
        stats = []
        for _ in range(n):
            pick = rng.choice(ug, len(ug))
            stats.append(f(np.concatenate([x[idx[g]] for g in pick])))
    return f(x), np.percentile(stats, 2.5), np.percentile(stats, 97.5)
