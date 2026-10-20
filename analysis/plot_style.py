"""Plot style of the paper figures, shared by the analysis notebooks.

    from plot_style import apply_style, COLORS, despine
    apply_style()

Rules (see CLAUDE.md, "Plotting"):
- axes start where the data starts (no padding) and end on a round tick value, as in the paper
  figures; quantities bounded by definition use their full range (q, probabilities: 0 to 1);
- no top/right spines, outward ticks, large sans-serif fonts;
- fixed colors per model / quantity (COLORS).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "pose_aware": "royalblue",     # Fig. 7 line color
    "no_pose": "forestgreen",
    "hist": "cornflowerblue",      # Fig. S1 histograms
    "diagonal": "red",             # y = x, dashed
    "good": "tab:green",           # pose-quality classes (shading)
    "bad": "tab:red",
    "neutral": "0.6",
}


def apply_style(font_scale=1.0):
    s = font_scale
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 13 * s,
        "axes.labelsize": 15 * s,
        "axes.titlesize": 14 * s,
        "xtick.labelsize": 12 * s,
        "ytick.labelsize": 12 * s,
        "legend.fontsize": 11 * s,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # axes start at the data and end on a round tick (paper figures: 0.0-1.0, 0-20000)
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.autolimit_mode": "round_numbers",
        "legend.frameon": True,
        "legend.fancybox": True,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def despine(ax, right=False):
    """Remove the top spine (and the right one unless a twin y axis needs it)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(right)
