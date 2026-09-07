"""Task 1 figure: effective calibration information over the transport splits.

Shows that the number of labelled target cells alone does not determine the
information available to estimate a calibration slope -- the figure the
manuscript refers to as \label{fig:information_spread}.
"""
from __future__ import annotations

import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14,
    "ytick.labelsize": 15, "legend.fontsize": 13, "axes.linewidth": 1.0,
})

QREI = Path(__file__).resolve().parent.parent
d = pd.read_csv(QREI / "results" / "t1_neff_transport_splits.csv")

MARK = {"chemistry": "o", "source": "s"}
COL = {
    "LFP": "#1f6f8b", "NCA": "#c0392b", "NMC": "#5b8c5a", "NMC/NCA": "#7d3c98",
    "BatteryLife SNL": "#e08a1e", "CALB": "#b03060", "HUST": "#2e86c1",
    "MATR": "#117a65", "SDU": "#d35400", "Tongji": "#566573",
}

fig, ax = plt.subplots(figsize=(9.4, 5.8))
ax.axhspan(1e-3, 1.0, color="0.85", alpha=0.55, zorder=0)
ax.axhline(1.0, color="0.45", ls=":", lw=1.4, zorder=1)
ax.axhline(4.0, color="0.45", ls="--", lw=1.4, zorder=1)

order = [("chemistry", g) for g in ["LFP", "NMC", "NCA", "NMC/NCA"]] +         [("source", g) for g in ["MATR", "Tongji", "HUST", "SDU",
                                 "BatteryLife SNL", "CALB"]]
handles = []
for ht, grp in order:
    g = d[(d.holdout_type == ht) & (d.holdout_group == grp)]
    if g.empty:
        continue
    ax.scatter(g.labelled_cells, g.n_eff_mean, s=95, marker=MARK[ht],
               facecolor=COL[grp], edgecolor="white", linewidth=1.0,
               alpha=0.95, zorder=3)
    handles.append(plt.Line2D([], [], marker=MARK[ht], ls="", ms=10,
                              markerfacecolor=COL[grp], markeredgecolor="white",
                              label="%s (%s)" % (grp, "chem" if ht == "chemistry" else "source")))

ax.set_yscale("log")
ax.set_xlabel("labelled target cells in the calibration subset")
ax.set_ylabel(r"effective calibration information  $n_{\mathrm{eff}}$")
ax.set_xlim(0, 76)
ax.set_ylim(4e-3, 13)
ax.text(1.5, 4.4, r"$n_{\mathrm{eff}}=4$   (slope SE 0.5)", fontsize=12.5,
        color="0.35")
ax.text(1.5, 0.70, r"$n_{\mathrm{eff}}=1$   (slope SE 1)", fontsize=12.5,
        color="0.35")
ax.grid(alpha=0.22, lw=0.6, which="both")
ax.legend(handles=handles, frameon=False, loc="lower right", ncol=2,
          handletextpad=0.3, columnspacing=1.0, labelspacing=0.35)

fig.tight_layout()
out = QREI / "figs" / "fig_transport_neff.png"
fig.savefig(out, dpi=320)
plt.close(fig)
print("wrote %s  (%d points)" % (out, len(d)))
