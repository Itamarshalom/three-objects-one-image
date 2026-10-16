"""Shared figure style: one colour per model, grouped by loss family."""
import matplotlib.pyplot as plt

MODELS = {
    "dav2-s": dict(label="Depth Anything V2-S", color="#9ecae1", family="L1"),
    "dav2-b": dict(label="Depth Anything V2-B", color="#4292c6", family="L1"),
    "dav2-l": dict(label="Depth Anything V2-L", color="#08519c", family="L1"),
    "dpt-l": dict(label="MiDaS DPT-L", color="#9e9ac8", family="L1"),
    "zoe": dict(label="ZoeDepth", color="#525252", family="logL2"),
    "lotus-d": dict(label="Lotus-D", color="#d94801", family="L2"),
    "lotus-g": dict(label="Lotus-G", color="#fd8d3c", family="L2"),
    "e2eft": dict(label="Marigold E2E-FT", color="#238b45", family="LSQ-L1"),
    "mg11": dict(label="Marigold v1.1", color="#c51b7d", family="gen"),
    "mg10": dict(label="Marigold v1.0", color="#e7298a", family="gen"),
}


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
    })


SHORT = {"dav2-s": "DA-S", "dav2-b": "DA-B", "dav2-l": "DA-L", "lotus-d": "Lotus-D", "lotus-g": "Lotus-G",
         "e2eft": "E2E-FT", "zoe": "ZoeDepth", "dpt-l": "MiDaS", "mg11-1": "Marig. v1.1", "mg10-1": "Marig. v1.0"}


def short(m):
    return SHORT.get(m, m)
