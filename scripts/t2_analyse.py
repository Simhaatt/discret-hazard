"""TASK 2 analysis -- items 2a to 2e plus the two Task 2 figures.

Reads t2_recal_<law>.json produced by t2_recal_boundary.py and writes
  results/t2_summary_<law>.json      every headline number
  results/t2_cells_<law>.csv         the per-cell table behind them
  figs/fig_crossover.png
  figs/fig_information_spread.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Figures carry no titles: the explanation belongs in the manuscript caption,
# not baked into the image. Type is sized for a one-column reduction.
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
})

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
S_EDGES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf]
N_EDGES = [0, 25, 50, 100, 200, 400]
ARMS = ["A", "B", "C", "D", "Dind", "E"]


def auc_of(score, label):
    """Rank-based AUC with mid-ranks for ties."""
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, float)
    s = score[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bin_table(df, key, edges, labels=None):
    idx = np.digitize(df[key].to_numpy(float), edges[1:-1], right=False)
    rows = []
    for b in range(len(edges) - 1):
        g = df[idx == b]
        lab = labels[b] if labels else (
            "[%g, %s)" % (edges[b], "inf" if not np.isfinite(edges[b + 1]) else "%g" % edges[b + 1])
        )
        if g.empty:
            rows.append(dict(bin=lab, cells=0))
            continue
        rows.append(dict(
            bin=lab, cells=int(len(g)),
            B_beats_A=int((g.LL_B < g.LL_A).sum()),
            E_beats_A=int((g.LL_E < g.LL_A).sum()),
            mean_LL_B_minus_A=float((g.LL_B - g.LL_A).mean()),
            mean_LL_E_minus_A=float((g.LL_E - g.LL_A).mean()),
            mean_n_eff=float(g.n_eff_mean.mean()),
        ))
    return pd.DataFrame(rows)


def analyse(law: str) -> dict:
    src = QREI / "results" / ("t2_recal_%s.json" % law)
    df = pd.DataFrame(json.load(open(src)))
    reps = int(df.reps.iloc[0])
    out: dict = dict(law=law, design_cells=int(len(df)), replicates_per_cell=reps,
                     total_replicates=int(len(df) * reps))

    # ---------------------------------------------------- 2a staging contributes
    dc = df.LL_D_minus_C
    hi = df[df.n >= 50]
    out["a_mean_LL_D_minus_C_all"] = float(dc.mean())
    out["a_mean_LL_D_minus_C_n_ge_50"] = float(hi.LL_D_minus_C.mean())
    out["a_max_abs_LL_D_minus_C"] = float(dc.abs().max())
    out["a_cells_exceeding_2mcse"] = int(
        (dc.abs() > 2.0 * df.LL_D_minus_C_se.replace(0.0, np.nan)).fillna(False).sum()
    )
    out["a_same_penalty_share_paired"] = float(df.same_penalty_CD.mean())
    out["a_same_penalty_share_paired_n_ge_50"] = float(hi.same_penalty_CD.mean())
    out["a_same_penalty_share_paired_n_25"] = float(df[df.n == 25].same_penalty_CD.mean())
    # the variant with an independent CV fold draw (CV noise retained)
    out["a_mean_LL_Dind_minus_C_all"] = float(df.LL_Dind_minus_C.mean())
    out["a_mean_LL_Dind_minus_C_n_ge_50"] = float(hi.LL_Dind_minus_C.mean())
    out["a_same_penalty_share_independent"] = float(df.same_penalty_CDind.mean())
    out["a_cells_Dind_exceeding_2mcse"] = int(
        (df.LL_Dind_minus_C.abs() > 2.0 * df.LL_Dind_minus_C_se.replace(0.0, np.nan))
        .fillna(False).sum()
    )

    # -------------------------------------------- 2b information spread at fixed n
    spread = (df.groupby("n")
                .agg(cells=("n_eff_mean", "size"),
                     min_n_eff=("n_eff_min", "min"),
                     max_n_eff=("n_eff_max", "max"),
                     min_cell_mean=("n_eff_mean", "min"),
                     max_cell_mean=("n_eff_mean", "max"))
                .reset_index())
    spread["fold_spread"] = spread.max_n_eff / spread.min_n_eff.replace(0.0, np.nan)
    spread["fold_spread_cell_means"] = spread.max_cell_mean / spread.min_cell_mean
    out["b_table"] = spread.to_dict("records")

    # ----------------------------------------- 2c where adaptation starts to pay
    out["c_table"] = bin_table(df, "S", S_EDGES).to_dict("records")

    # ------------------------------------------------ 2d n vs n_eff as criterion
    label = (df.LL_B < df.LL_A).astype(int).to_numpy()
    out["d_auc_n"] = auc_of(df.n.to_numpy(float), label)
    out["d_auc_n_eff"] = auc_of(df.n_eff_mean.to_numpy(float), label)
    out["d_auc_S"] = auc_of(df.S.to_numpy(float), label)
    out["d_positives"] = int(label.sum())
    out["d_negatives"] = int((1 - label).sum())
    sub = df[df.b0 != 1.0]
    lab2 = (sub.LL_B < sub.LL_A).astype(int).to_numpy()
    out["d_auc_n_b0_ne_1"] = auc_of(sub.n.to_numpy(float), lab2)
    out["d_auc_n_eff_b0_ne_1"] = auc_of(sub.n_eff_mean.to_numpy(float), lab2)
    out["d_positives_b0_ne_1"] = int(lab2.sum())
    out["d_negatives_b0_ne_1"] = int((1 - lab2).sum())
    nlab = ["n = %d" % v for v in sorted(df.n.unique())]
    out["d_table_by_n"] = bin_table(df, "n", [0, 26, 51, 101, 201, np.inf], nlab).to_dict("records")

    # -------------------------------------------------------- 2e risk envelope
    env = []
    for k in ["A", "B", "C", "D", "Dind", "E"]:
        ex = df["LL_" + k] - df.LL_oracle
        env.append(dict(arm=k, mean_excess=float(ex.mean()), max_excess=float(ex.max()),
                        median_excess=float(ex.median()),
                        cells_within_0p001=int((ex <= 1e-3).sum())))
    out["e_table"] = env

    # -------------------------------------------------------------- diagnostics
    out["median_mcse_LL"] = {k: float(df["LLse_" + k].median()) for k in ARMS}
    out["max_mcse_LL"] = {k: float(df["LLse_" + k].max()) for k in ARMS}
    out["median_sd_slope"] = {k: float(df["sdb_" + k].median()) for k in ARMS}
    out["mean_fix_b1_share"] = {k: float(df["fix_share_" + k].mean()) for k in ARMS}
    out["mean_one_class_share"] = float(df.one_class_share.mean())
    out["max_one_class_share"] = float(df.one_class_share.max())
    out["one_class_share_by_n"] = df.groupby("n").one_class_share.mean().to_dict()
    out["mean_ml_nonconvergence"] = float(df.ml_nonconvergence_share.mean())
    out["n_eff_overall_range"] = [float(df.n_eff_min.min()), float(df.n_eff_max.max())]

    df.to_csv(QREI / "results" / ("t2_cells_%s.csv" % law), index=False)
    (QREI / "results" / ("t2_summary_%s.json" % law)).write_text(json.dumps(out, indent=1, default=float))
    return out


def figures(law: str) -> None:
    df = pd.read_csv(QREI / "results" / ("t2_cells_%s.csv" % law))
    figs = QREI / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    # --- fig_crossover
    idx = np.digitize(df.S.to_numpy(float), S_EDGES[1:-1], right=False)
    centres, mb, me, nb = [], [], [], []
    labels = []
    for b in range(len(S_EDGES) - 1):
        g = df[idx == b]
        if g.empty:
            continue
        centres.append(b)
        labels.append("[%g,%s)" % (S_EDGES[b],
                                   "inf" if not np.isfinite(S_EDGES[b + 1]) else "%g" % S_EDGES[b + 1]))
        mb.append(float((g.LL_B - g.LL_A).mean()))
        me.append(float((g.LL_E - g.LL_A).mean()))
        nb.append(len(g))
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.axhline(0.0, color="0.35", lw=1.2, ls="--", zorder=1)
    ax.plot(centres, mb, "o-", color="#c0392b", lw=2.4, ms=9,
            label="B  unrestricted two-parameter ML")
    ax.plot(centres, me, "s-", color="#1f6f8b", lw=2.4, ms=9,
            label="E  information-adaptive")
    for x, y, k in zip(centres, mb, nb):
        ax.annotate("%d" % k, (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=12, color="#c0392b")
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("slope-shift signal  "
                  r"$S=(b_0-1)^2\,n_{\mathrm{eff}}$  (bin)")
    ax.set_ylabel(r"mean expected $\mathrm{LL}(\mathrm{arm})-\mathrm{LL}(A)$")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(figs / "fig_crossover.png", dpi=320)
    plt.close(fig)

    # --- fig_information_spread
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for n, g in df.groupby("n"):
        lo, hi = float(g.n_eff_min.min()), float(g.n_eff_max.max())
        ax.plot([lo, hi], [n, n], color="#1f6f8b", lw=10, alpha=0.35,
                solid_capstyle="butt")
        ax.plot(g.n_eff_mean, np.full(len(g), n), "|", color="#c0392b", ms=16, mew=1.6)
        ax.annotate("%.0f-fold" % (hi / lo if lo > 0 else np.nan), (hi, n),
                    textcoords="offset points", xytext=(12, 0), va="center",
                    fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks(sorted(df.n.unique()))
    ax.set_yticklabels([str(v) for v in sorted(df.n.unique())])
    ax.set_xlabel("effective calibration information  "
                  r"$n_{\mathrm{eff}}$")
    ax.set_ylabel("labelled calibration units  $n$")
    ax.axvline(1.0, color="0.5", ls=":", lw=1.4)
    ax.axvline(4.0, color="0.5", ls="--", lw=1.4)
    ax.set_xlim(right=ax.get_xlim()[1] * 3.2)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    fig.tight_layout()
    fig.savefig(figs / "fig_information_spread.png", dpi=320)
    plt.close(fig)
    print("wrote %s and %s" % (figs / "fig_crossover.png", figs / "fig_information_spread.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", default="project")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    out = analyse(args.law)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))},
                     indent=1, default=float))
    if not args.no_figs and args.law == "project":
        figures(args.law)


if __name__ == "__main__":
    main()
