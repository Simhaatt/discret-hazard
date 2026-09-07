"""TASK 4 analysis -- Table 4 and fig_firstpassage.png."""
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

# No title on the figure: that belongs in the manuscript caption.
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
})

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
ORDER = ["wiener", "gamma", "cloglog hazard model", "random forest",
         "cloglog cumulative-risk (artefact-affected)"]
COLOUR = {"wiener": "#c0392b", "gamma": "#e08a1e",
          "cloglog hazard model": "#1f6f8b", "random forest": "#5b8c5a",
          "cloglog cumulative-risk (artefact-affected)": "#999999"}
MARK = {"wiener": "o", "gamma": "s", "cloglog hazard model": "^",
        "random forest": "D", "cloglog cumulative-risk (artefact-affected)": "x"}


def load():
    frames = []
    for tag in ["", "_matr", "_pool"]:
        p = QREI / "results" / ("t4_first_passage_metrics%s.csv" % tag)
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("no Task 4 metrics found")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(
        subset=["corpus", "timescale", "variant", "window", "landmark_cycle",
                "horizon_cycle", "model"], keep="last")


def table4(df, corpus="matr", timescale="cycle", variant="fair", window="full"):
    sel = df[(df.corpus == corpus) & (df.timescale == timescale)
             & (df.variant == variant) & (df.window == window)].copy()
    sel["model"] = pd.Categorical(sel.model, [m for m in ORDER if m in set(sel.model)],
                                  ordered=True)
    sel = sel.sort_values(["model", "landmark_cycle", "horizon_cycle"])
    return sel


def figure(df):
    figs = QREI / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    sel = table4(df)
    if sel.empty:
        print("skipping fig_firstpassage: no MATR rows")
        return
    settings = (sel[["landmark_cycle", "horizon_cycle"]].drop_duplicates()
                .sort_values(["landmark_cycle", "horizon_cycle"]))
    labels = ["L%d/H%d" % (r.landmark_cycle, r.horizon_cycle)
              for r in settings.itertuples()]
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for model in [m for m in ORDER if m in set(sel.model)]:
        g = sel[sel.model == model].set_index(["landmark_cycle", "horizon_cycle"])
        ys = [g.loc[(r.landmark_cycle, r.horizon_cycle), "brier"]
              if (r.landmark_cycle, r.horizon_cycle) in g.index else np.nan
              for r in settings.itertuples()]
        ax.plot(xs, ys, marker=MARK[model], color=COLOUR[model], lw=2.4, ms=9,
                label=model)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("landmark / horizon setting (cycles)")
    ax.set_ylabel("Brier score")
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.25, lw=0.6)
    ax.margins(y=0.10)
    fig.tight_layout()
    fig.savefig(figs / "fig_firstpassage.png", dpi=320)
    plt.close(fig)
    print("wrote %s" % (figs / "fig_firstpassage.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()
    df = load()
    pd.set_option("display.width", 220)

    out = {}
    print("=== Table 4 (MATR, raw cycle, req-4 enforced, full train window) ===")
    t = table4(df)
    cols = ["model", "landmark_cycle", "horizon_cycle", "n_evaluated", "n_events",
            "brier", "log_loss", "auc", "mean_pred", "observed_rate"]
    print(t[cols].to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\nmeans across the nine settings:")
    means = t.groupby("model", observed=True)[["brier", "log_loss", "auc", "mean_pred",
                                               "observed_rate"]].mean()
    print(means.round(4).to_string())
    out["table4_matr"] = t[cols].to_dict("records")
    out["table4_matr_means"] = means.round(6).reset_index().to_dict("records")

    print("\n=== sensitivity: window / variant / timescale (MATR, mean Brier) ===")
    piv = (df[df.corpus == "matr"]
           .pivot_table(index="model", columns=["window", "variant", "timescale"],
                        values="brier", aggfunc="mean"))
    print(piv.round(4).to_string())
    out["matr_sensitivity_mean_brier"] = json.loads(piv.round(6).to_json())

    # provable invariance of a pure rescaling of time for a single-source corpus
    a = df[(df.corpus == "matr") & (df.timescale == "cycle") & (df.variant == "fair")
           & (df.window == "full")].set_index(["model", "landmark_cycle", "horizon_cycle"])
    b = df[(df.corpus == "matr") & (df.timescale == "normalised") & (df.variant == "fair")
           & (df.window == "full")].set_index(["model", "landmark_cycle", "horizon_cycle"])
    j = a.join(b, rsuffix="_n", how="inner")
    if len(j):
        d = (j.brier - j.brier_n).abs().max()
        print("\nraw cycle vs per-source-normalised time, MATR: max |Brier diff| = %.3e" % d)
        out["matr_timescale_invariance_max_brier_diff"] = float(d)

    if (df.corpus == "sixsource").any():
        print("\n=== six-source corpus (five-fold whole-cell CV, mean over settings) ===")
        s = df[df.corpus == "sixsource"]
        pv = s.pivot_table(index="model", columns=["window", "timescale"],
                           values=["brier", "auc"], aggfunc="mean")
        print(pv.round(4).to_string())
        out["sixsource_means"] = json.loads(pv.round(6).to_json())
        print("\nper-setting detail (cycle scale, full window):")
        det = s[(s.timescale == "cycle") & (s.window == "full")]
        print(det[cols].to_string(index=False, float_format=lambda v: "%.4f" % v))
        out["sixsource_detail_cycle_full"] = det[cols].to_dict("records")

    # fitted parameters, physical plausibility
    pf = []
    for tag in ["", "_matr", "_pool"]:
        p = QREI / "results" / ("t4_first_passage_params%s.csv" % tag)
        if p.exists():
            pf.append(pd.read_csv(p))
    if pf:
        par = pd.concat(pf, ignore_index=True)
        print("\n=== fitted parameters, physical plausibility ===")
        w = par[(par.model == "wiener") & (par.timescale == "cycle")
                & (par.variant == "fair")].copy()
        w["cycles_to_threshold"] = 0.20 / w.mu0
        g = par[(par.model == "gamma") & (par.timescale == "cycle")
                & (par.variant == "fair")].copy()
        g["mean_rate"] = g.alpha * g.b0 / g.a0
        g["cycles_to_threshold"] = 0.20 / g.mean_rate
        for nm, fr, keys in [("wiener", w, ["mu0", "s2mu", "sig2", "cycles_to_threshold", "converged"]),
                             ("gamma", g, ["alpha", "a0", "b0", "mean_rate", "cycles_to_threshold", "converged"])]:
            print("\n%s (raw cycle scale, req-4):" % nm)
            print(fr.groupby(["corpus", "window"])[keys[:-1]].mean().to_string(
                float_format=lambda v: "%.6g" % v))
            print("  converged share: %s"
                  % fr.groupby(["corpus", "window"]).converged.mean().round(3).to_dict())
        out["wiener_params"] = w.groupby(["corpus", "window"])[
            ["mu0", "s2mu", "sig2", "cycles_to_threshold"]].mean().reset_index().to_dict("records")
        out["gamma_params"] = g.groupby(["corpus", "window"])[
            ["alpha", "a0", "b0", "mean_rate", "cycles_to_threshold"]].mean().reset_index().to_dict("records")

    (QREI / "results" / "t4_summary.json").write_text(json.dumps(out, indent=1, default=float))
    if not args.no_figs:
        figure(df)


if __name__ == "__main__":
    main()
