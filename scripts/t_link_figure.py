"""Figure 2 -- link bias and empirical coverage after temporal grouping.

Two panels over interval width 0.5, 2, 5:
  (a) mean coefficient bias, reference line at zero
  (b) empirical 95% coverage, reference line at the nominal 0.95

Values are read from results/link_simulation_bias_coverage.csv, which is the
project's own archived Study-A link table (table01_link_grouping_simulation),
18 design cells averaged per point at 1,000 replicates each -- they are not
retyped into this script.
"""
from __future__ import annotations

import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 15,
    "ytick.labelsize": 15, "legend.fontsize": 15, "axes.linewidth": 1.0,
})

QREI = Path(__file__).resolve().parent.parent
d = pd.read_csv(QREI / "results" / "link_simulation_bias_coverage.csv")

LINKS = [("cloglog", "complementary log-log", "#1f6f8b", "o"),
         ("logit", "logit", "#c0392b", "s"),
         ("probit", "probit", "#e08a1e", "D")]
widths = sorted(d.width.unique())
xs = range(len(widths))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2))

# ---- (a) bias
ax1.axhline(0.0, color="0.35", ls="--", lw=1.4, zorder=1)
for key, lab, col, mk in LINKS:
    g = d[d.link == key].set_index("width").loc[widths]
    ax1.plot(xs, g.bias, marker=mk, color=col, lw=2.6, ms=10, label=lab, zorder=3)
ax1.set_xticks(list(xs))
ax1.set_xticklabels([("%g" % w) for w in widths])
ax1.set_xlabel("interval width")
ax1.set_ylabel("mean coefficient bias")
ax1.set_xlim(-0.28, len(widths) - 0.72)
ax1.set_ylim(-0.36, 0.20)
ax1.text(len(widths) - 0.78, 0.012, "unbiased", fontsize=13.5,
         color="0.35", ha="right", va="bottom",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
ax1.grid(alpha=0.22, lw=0.6)
ax1.set_title("(a)  coefficient bias", fontsize=16, loc="left", pad=10)

# ---- (b) coverage
ax2.axhline(0.95, color="0.35", ls="--", lw=1.4, zorder=1)
for key, lab, col, mk in LINKS:
    g = d[d.link == key].set_index("width").loc[widths]
    ax2.plot(xs, g.coverage, marker=mk, color=col, lw=2.6, ms=10, label=lab,
             zorder=3)
ax2.set_xticks(list(xs))
ax2.set_xticklabels([("%g" % w) for w in widths])
ax2.set_xlabel("interval width")
ax2.set_ylabel("empirical 95% coverage")
ax2.set_xlim(-0.28, len(widths) - 0.72)
ax2.set_ylim(-0.05, 1.05)
ax2.text(len(widths) - 0.78, 0.865, "nominal 0.95", fontsize=13.5,
         color="0.35", ha="right", va="top",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
ax2.grid(alpha=0.22, lw=0.6)
ax2.set_title("(b)  empirical coverage", fontsize=16, loc="left", pad=10)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.01), columnspacing=2.4, handletextpad=0.5)

fig.tight_layout(rect=(0, 0.085, 1, 1))
out = QREI / "figs" / "fig_link_grouping.png"
fig.savefig(out, dpi=320)
plt.close(fig)
print("wrote %s" % out)
for key, lab, _, _ in LINKS:
    g = d[d.link == key].set_index("width").loc[widths]
    print("  %-8s bias %s | coverage %s"
          % (key, " ".join("%+.4f" % v for v in g.bias),
             " ".join("%.4f" % v for v in g.coverage)))
