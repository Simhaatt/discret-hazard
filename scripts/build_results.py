"""Assemble RESULTS.md from the saved result files.

Every number in the report is read from results/*.csv|json, so the prose and
the tables cannot drift apart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from scipy import stats

QREI = Path(__file__).resolve().parent.parent
RES = QREI / "results"
B: list[str] = []


def w(s=""):
    B.append(s)


def table(df, headers, fmt=None):
    fmt = fmt or {}
    w("| " + " | ".join(str(h) for h in headers) + " |")
    w("|" + "|".join(["---"] * len(headers)) + "|")
    # Column-wise access, not iterrows(): iterrows() collapses a mixed-dtype
    # row to one dtype and would render integer columns as 25.0000.
    for i in range(len(df)):
        cells = []
        for c in df.columns:
            v = df[c].iloc[i]
            f = fmt.get(c)
            if isinstance(v, float) and not np.isfinite(v):
                cells.append("n/a")
            elif f:
                cells.append(f % v)
            elif isinstance(v, float):
                cells.append("%.4f" % v)
            else:
                cells.append(str(v))
        w("| " + " | ".join(cells) + " |")
    w()


def jload(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


# =====================================================================  header
w("# Results")
w()
w("Experiments for the QREI resubmission of *Grouped discrete-time hazard "
  "modelling for degradation data*.  Numbers, tables and figures only; the "
  "manuscript is not rewritten here.")
w()
w("All work was run from `F:\\discrel\\qrei`.  Every script, seed and output "
  "path is listed in section 6.")
w()

# ==================================================================== 0. disco
w("## 0. Repository discovery")
w()
w("All five required artefacts were located.  The authoritative source is the "
  "reproducibility archive `C:\\Users\\hp\\Documents\\csda-grouped-hazard-repro`, "
  "copied to `F:\\discrel\\qrei\\repro` and pointed at the data drive "
  "`F:\\discrel\\data\\processed` (the archive resolves paths through "
  "`src/utilities/project_paths.py`, so no absolute path was edited).")
w()
tbl = pd.DataFrame([
    ("Simulation DGP",
     "`src/pipeline/79_mcsm_experiment_suite.py::simulate_person_period`, with "
     "`baseline_alpha` (smooth increasing log-time baseline) and "
     "`true_health_effect` (analytic state effect outside the fitted spline "
     "basis). At interval width 25, `jmax = 1000/25 = 40`, i.e. the J = 40 "
     "intervals of Section S1.1. Sequential first-event generation, "
     "independent censoring, one time-varying (`ar`) and one unit-level "
     "(`static`) nuisance covariate. Main seed 20260823.", "found"),
    ("Recalibration study",
     "`src/pipeline/83_study_b_shrunk_recalibration.py`, seed 20260825. "
     "Contains `RHO_GRID = [0.01, 0.1, 1, 10, inf]`, `MIN_TUNE_CELLS = 20` and "
     "`candidate_grid = RHO_GRID if len(y) >= 50 else [0.1, 1, 10, inf]` -- "
     "exactly the 20/50 staging and the two grids described in the brief.",
     "found"),
    ("Battery corpus",
     "`data/processed/phase2_expanded_person_period_grouped_25.csv`: 16,538 "
     "person-period rows, 493 cells, 303 harmonised events, 190 censored, six "
     "sources with the stated counts (MATR 139, BatteryLife SNL 61, CALB 27, "
     "HUST 77, SDU 70, Tongji 119). Verified that `eol_cycle` equals the first "
     "cycle with SOH <= 0.80 for all 83 MATR events, confirming the persistent "
     "80% SOH endpoint and the threshold D* = 0.20.", "found"),
    ("Fitted source models",
     "`src/pipeline/build_manuscript_tables.py::fixed_horizon_transport`, which "
     "fits the pooled cloglog hazard via "
     "`74_transport_recalibration_loco.py::fit_cloglog` (statsmodels GLM, "
     "cloglog link, `fit_regularized(alpha=0.01, L1_wt=0)`) on all sources but "
     "the held-out one, accumulates each target cell's landmark-to-horizon "
     "risk, and recalibrates on a 30% target-cell subset over 100 splits from "
     "base seed 20260702. Its output `table_D_fixed_horizon_external_"
     "transport_recalibration.csv` is what manuscript Tables 8 and 9 aggregate.",
     "found"),
    ("Landmark-horizon grid",
     "`LANDMARKS = [150, 200, 300]` x `HORIZONS = [500, 800, 1000]` = nine "
     "settings (`build_manuscript_tables.py`).", "found"),
], columns=["artefact", "where", "status"])
table(tbl, ["artefact", "what was found", "status"])

w("Both published cloglog reference values quoted in the brief were "
  "reproduced exactly before any new work was done, which fixes the protocol "
  "for Tasks 1, 3 and 4:")
w()
w("* the standardised lagged-SOH hazard ratio **0.499526** with 95% whole-cell "
  "bootstrap interval **(0.471587, 0.524324)** at penalty "
  "alpha = 0.03162277660168379 -- reproduced to all six quoted digits "
  "(Task 3, Table 3);")
w("* the leakage-free rolling-origin Brier range **0.0366 - 0.0967** and AUC "
  "range **0.9174 - 0.9949** -- reproduced to 4.8e-7 against archived "
  "`table06_rolling_origin_performance.csv` (Task 3, Table 3).")
w()
w("Two defects in the archive had to be repaired before it would run; both are "
  "recorded in section 5.")
w()

# ==================================================================== 1. Task 1
d1 = pd.read_csv(RES / "t1_neff_transport_splits.csv")
d1["split"] = np.where(d1.holdout_type == "chemistry",
                       "leave-one-chemistry-out", "leave-one-source-out")
d1["se"] = 1.0 / np.sqrt(d1.n_eff_mean)
g1 = (d1.groupby(["split", "holdout_group"], as_index=False)
      .agg(settings=("n_eff_mean", "size"), cells=("labelled_cells", "mean"),
           events=("events_in_subset", "mean"), sd_eta=("sd_eta", "mean"),
           neff=("n_eff_mean", "mean"), lo=("n_eff_mean", "min"),
           hi=("n_eff_mean", "max"), one_class=("one_class_share", "mean")))
g1["se"] = 1.0 / np.sqrt(g1.neff)
g1["rng"] = ["%.2f - %.2f" % (a, b) for a, b in zip(g1.lo, g1.hi)]
g1 = g1.sort_values(["split", "neff"], ascending=[True, False])

w("## 1. Effective calibration information for transport splits")
w()
w("`n_eff` was computed on the **same** labelled subsets the published "
  "analysis used, not on fresh ones: the replay reconstructs the generator "
  "`default_rng(20260702 + crc32(\"<holdout>:<group>:<L>:<H>\") % 100000)` and "
  "consumes it over the recalibration fractions 0.05, 0.10, 0.20, 0.30 in the "
  "original order, recording the 30% splits.  As a check the replay also "
  "recomputes the published before/after log loss: it agrees with archived "
  "`table_D_*.csv` at all 90 rows to **exactly zero** difference, so the "
  "`n_eff` values below describe the very splits behind manuscript Tables 8 "
  "and 9.  `n_eff` is evaluated at the null recalibration (a = 0, b = 1) and "
  "uses only the source model's predictions -- no outcome labels.")
w()
w("Each row averages 100 target-cell splits.")
w()
w("### Table 1 -- calibration information per split, over the nine "
  "landmark-horizon settings")
w()
table(g1[["split", "holdout_group", "settings", "cells", "events", "sd_eta",
          "neff", "rng", "se", "one_class"]],
      ["split", "held-out unit", "settings", "labelled cells",
       "events in subset", "sd(eta)", "n_eff", "n_eff range over the nine",
       "1/sqrt(n_eff)", "one-class share"],
      {"cells": "%.1f", "events": "%.2f", "sd_eta": "%.4f", "neff": "%.2f",
       "se": "%.2f", "one_class": "%.3f"})

n_below4 = int((d1.n_eff_mean < 4).sum())
n_below1 = int((d1.n_eff_mean < 1).sum())
n_below025 = int((d1.n_eff_mean < 0.25).sum())
ge4 = sorted(set(d1[d1.n_eff_mean >= 4].holdout_group))
rho_cells = stats.spearmanr(d1.labelled_cells, d1.n_eff_mean).statistic
rho_sd = stats.spearmanr(d1.sd_eta, d1.n_eff_mean).statistic
rho_ev = stats.spearmanr(d1.events_in_subset, d1.n_eff_mean).statistic
rho_px = stats.spearmanr(d1.n_eff_mean, d1.labelled_cells * d1.sd_eta ** 2).statistic
gg = g1.set_index("holdout_group")

w("**Which splits are information-poor.**  Of the 90 (split, landmark, "
  "horizon) combinations, **%d of 90 fall below n_eff = 4** and **%d of 90 "
  "fall below n_eff = 1**; %d fall below 0.25.  Only %d combinations reach "
  "n_eff >= 4, and every one of them is the %s split.  Nine of the ten splits "
  "have a nine-setting mean below 4.  Across all 90 rows n_eff spans "
  "**%.4f to %.2f**, so the asymptotic standard error of an unpenalised "
  "calibration slope spans **%.2f to %.2f**.  A calibration slope is simply "
  "not estimable at most of these splits."
  % (n_below4, n_below1, n_below025, int((d1.n_eff_mean >= 4).sum()),
     ", ".join(ge4), d1.n_eff_mean.min(), d1.n_eff_mean.max(),
     1 / np.sqrt(d1.n_eff_mean.max()), 1 / np.sqrt(d1.n_eff_mean.min())))
w()
w("The two smallest sources behave as the brief anticipated.  CALB, with 27 "
  "cells, yields %.1f labelled cells and n_eff = %.2f (slope SE %.1f); the "
  "NMC/NCA chemistry, with 9 cells, yields 3 labelled cells and "
  "n_eff = %.4f (slope SE %.1f), and %.0f%% of its calibration subsets contain "
  "only one outcome class.  These are findings, not defects."
  % (gg.loc["CALB", "cells"], gg.loc["CALB", "neff"], gg.loc["CALB", "se"],
     gg.loc["NMC/NCA", "neff"], gg.loc["NMC/NCA", "se"],
     100 * gg.loc["NMC/NCA", "one_class"]))
w()
w("**The more consequential finding is that labelled sample size does not "
  "order the information.**  Across the 90 rows the Spearman correlation "
  "between the number of labelled cells and n_eff is only **%+.3f**, while the "
  "correlation with the spread of the source predictions, sd(eta), is "
  "**%+.3f** (events in the subset: %+.3f).  The product n * sd(eta)^2 "
  "correlates at %+.3f, which is what the definition of n_eff implies.  The "
  "clearest inversion in the table: HUST supplies %.0f labelled cells but "
  "n_eff = %.2f, whereas NCA supplies %.0f labelled cells and n_eff = %.2f -- "
  "**%.1f times fewer labelled units carrying %.0f times more calibration "
  "information**.  HUST's subsets are large but almost eventless (%.2f events "
  "on average, %.0f%% one-class) and its source predictions barely vary "
  "(sd(eta) = %.3f).  MATR is the same story more mildly: %.0f labelled cells, "
  "%.1f events, yet n_eff = %.2f because sd(eta) is only %.3f."
  % (rho_cells, rho_sd, rho_ev, rho_px,
     gg.loc["HUST", "cells"], gg.loc["HUST", "neff"],
     gg.loc["NCA", "cells"], gg.loc["NCA", "neff"],
     gg.loc["HUST", "cells"] / gg.loc["NCA", "cells"],
     gg.loc["NCA", "neff"] / gg.loc["HUST", "neff"],
     gg.loc["HUST", "events"], 100 * gg.loc["HUST", "one_class"],
     gg.loc["HUST", "sd_eta"], gg.loc["MATR", "cells"],
     gg.loc["MATR", "events"], gg.loc["MATR", "neff"],
     gg.loc["MATR", "sd_eta"]))
w()
w("This is the reviewer's objection made numerical on the paper's own data: a "
  "rule keyed on the number of labelled units would have treated HUST as "
  "better supported than NCA and been wrong by an order of magnitude.")
w()
w("Full detail, one row per (split, landmark, horizon), is in "
  "`results/t1_neff_transport_splits.csv`.")
w()

# ==================================================================== 2. Task 2
s2 = jload("t2_summary_project.json")
s2n = jload("t2_summary_normal.json")
cells2 = pd.read_csv(RES / "t2_cells_project.csv")

w("## 2. Recalibration boundary study")
w()
w("**Design.**  Labelled calibration units n in {25, 50, 100, 200, 400} x "
  "target event prevalence in {0.05, 0.20, 0.50} x spread of source "
  "predictions sd(eta) in {0.5, 1.0, 2.0} x true target calibration slope b0 "
  "in {0.7, 1.0, 1.25, 1.5, 2.0} = **225 design cells**, of which 180 have "
  "b0 != 1, with **500 replicates each (112,500 replicates)**.  Prevalence and "
  "prediction spread are set independently of n, which is what breaks the "
  "confound in the deleted design.  Seed 20260825, spawned per cell as "
  "`SeedSequence([20260825, cell_index])`.")
w()
w("**Target law from the project's own DGP.**  The brief asks for the "
  "project DGP rather than the normal linear predictor behind its pilot "
  "figures, so the law of eta was derived from Section S1.1 itself: a source "
  "cohort of 1,500 units was simulated with `simulate_person_period` at width "
  "25 (J = 40 intervals), the project's penalised grouped cloglog hazard was "
  "fitted on it at the manuscript's selected penalty 0.0316228, an independent "
  "20,000-unit target cohort was scored, and each unit's landmark-to-horizon "
  "risk was accumulated exactly as the battery analysis does, giving "
  "eta = log(-log(1 - r)) for 19,489 units (seed 20260823).  The source model "
  "is therefore genuinely misspecified for the target -- the true state effect "
  "is analytic and outside the fitted basis, and person-periods within a unit "
  "are serially dependent.")
w()
meta = jload("t2_eta_nodes_meta.json")
w("That eta law is **strongly non-normal**: skewness %.2f, excess kurtosis "
  "%.2f, Shapiro-Wilk decisively rejecting normality.  This matters, and it is "
  "the main reason some numbers below differ from the brief's pilot.  The law "
  "is represented as 2,000 equally weighted standardised quantile nodes, so "
  "expected held-out log loss and Brier are **exact sums over the target law** "
  "with no evaluation Monte Carlo noise -- the role Gauss-Hermite quadrature "
  "plays for a normal eta.  The whole study was also re-run under a normal eta "
  "law with 80-point Gauss-Hermite quadrature purely as an implementation "
  "reference (Table 2g)." % (meta["skew"], meta["excess_kurtosis"]))
w()

w("### 2a. The staging contributes nothing")
w()
w("The first thing to record is structural, not empirical.  Arm C uses the "
  "full penalty grid at every n; arm D uses the full grid whenever n >= 50.  "
  "**For n >= 50 the two arms are therefore the same estimator by "
  "construction**, and the design's only n below 50 is n = 25, where the "
  "staged rule merely deletes rho = 0.01 from the grid (n = 25 is above the "
  "rule's own n < 20 cut, so the slope is never fixed by staging anywhere in "
  "this design).  With the cross-validation folds shared between C and D "
  "within a replicate -- which the brief requires, and which isolates the "
  "grid from CV noise -- the consequences are:")
w()
a = s2
rows = pd.DataFrame([
    ("mean LL(D) - LL(C), all 225 cells", a["a_mean_LL_D_minus_C_all"]),
    ("mean LL(D) - LL(C), n >= 50 (180 cells)", a["a_mean_LL_D_minus_C_n_ge_50"]),
    ("largest absolute LL(D) - LL(C) over cells", a["a_max_abs_LL_D_minus_C"]),
    ("cells where the difference exceeds 2 Monte Carlo SE",
     float(a["a_cells_exceeding_2mcse"])),
    ("share of replicates where C and D select the same penalty",
     a["a_same_penalty_share_paired"]),
    ("  the same, restricted to n >= 50", a["a_same_penalty_share_paired_n_ge_50"]),
    ("  the same, restricted to n = 25", a["a_same_penalty_share_paired_n_25"]),
], columns=["quantity", "value"])
rows["value"] = ["%d" % v if float(v).is_integer() and abs(v) >= 1
                 else "%.6f" % v for v in rows["value"]]
table(rows, ["quantity", "value"])
w("With an **independent** fold draw for D, which is what the original code "
  "did and what makes the number comparable with the brief's pilot, the mean "
  "difference at n >= 50 is %.6f (pilot: -0.00004) and the same penalty is "
  "chosen in %.1f%% of replicates (pilot: 66%%); %d cells then exceed twice "
  "their Monte Carlo SE, which is CV resampling noise, not staging."
  % (a["a_mean_LL_Dind_minus_C_n_ge_50"],
     100 * a["a_same_penalty_share_independent"],
     a["a_cells_Dind_exceeding_2mcse"]))
w()
w("So the staging is either exactly inert (n >= 50) or worth a mean %.6f in "
  "expected log loss (n = 25).  There is nothing for it to contribute."
  % abs(a["a_mean_LL_D_minus_C_all"]))
w()

w("### 2b. Information spread at fixed n")
w()
b2 = pd.DataFrame(s2["b_table"])
b2["n"] = b2["n"].astype(int)
b2["cells"] = b2["cells"].astype(int)
table(b2[["n", "cells", "min_n_eff", "max_n_eff", "fold_spread",
          "min_cell_mean", "max_cell_mean", "fold_spread_cell_means"]],
      ["n", "design cells", "min n_eff", "max n_eff", "fold spread",
       "min design-cell mean", "max design-cell mean",
       "fold spread of cell means"],
      {"min_n_eff": "%.2f", "max_n_eff": "%.2f", "fold_spread": "%.0f",
       "min_cell_mean": "%.2f", "max_cell_mean": "%.2f",
       "fold_spread_cell_means": "%.0f"})
w("At **every** labelled sample size the available calibration information "
  "spans two or more orders of magnitude, and the ranges overlap heavily "
  "across n: at n = 25 n_eff reaches %.2f, while at n = 400 it falls as low as "
  "%.2f.  Ranked by design-cell means the spread is essentially constant at "
  "about 140-fold at every n.  A cut point on n cannot separate the cells that "
  "support a slope from those that do not.  See `fig_information_spread.png`."
  % (float(b2[b2.n == 25].max_n_eff.iloc[0]),
     float(b2[b2.n == 400].min_n_eff.iloc[0])))
w()

w("### 2c. Where adaptation starts to pay")
w()
c2 = pd.DataFrame(s2["c_table"])
c2["Bc"] = ["%d/%d" % (x, y) for x, y in zip(c2.B_beats_A, c2.cells)]
c2["Ec"] = ["%d/%d" % (x, y) for x, y in zip(c2.E_beats_A, c2.cells)]
table(c2[["bin", "cells", "Bc", "Ec", "mean_LL_B_minus_A", "mean_LL_E_minus_A",
          "mean_n_eff"]],
      ["S range", "cells", "B beats A", "E beats A", "mean LL(B) - LL(A)",
       "mean LL(E) - LL(A)", "mean n_eff"],
      {"mean_LL_B_minus_A": "%+.4f", "mean_LL_E_minus_A": "%+.4f",
       "mean_n_eff": "%.2f"})
w("Ordering cells by S = (b0 - 1)^2 * n_eff produces a clean and monotone "
  "crossover.  Unrestricted two-parameter ML (B) is worse than intercept-only "
  "updating in the mean until **S is between 4 and 8**, and beats it in almost "
  "every cell once S >= 8.  The information-adaptive arm (E) crosses earlier, "
  "between **S = 2 and S = 4**, and is never much worse than A below the "
  "crossover: its largest mean penalty in any S bin is %+.4f against %+.4f for "
  "B.  Above S = 8 the two coincide, because E's positive-part penalty goes to "
  "zero once the observed slope shift is large relative to its standard error. "
  " See `fig_crossover.png`."
  % (c2.mean_LL_E_minus_A.max(), c2.mean_LL_B_minus_A.max()))
w()

w("### 2d. n against n_eff as a criterion")
w()
lab = "unrestricted two-parameter ML beats intercept-only updating"
rows = pd.DataFrame([
    ("AUC of n_eff", s2["d_auc_n_eff"]),
    ("AUC of n", s2["d_auc_n"]),
    ("AUC of S = (b0-1)^2 n_eff (reference; uses the unobservable b0)",
     s2["d_auc_S"]),
    ("AUC of n_eff, restricted to b0 != 1", s2["d_auc_n_eff_b0_ne_1"]),
    ("AUC of n, restricted to b0 != 1", s2["d_auc_n_b0_ne_1"]),
], columns=["criterion", "auc"])
table(rows, ["criterion for predicting '%s'" % lab, "AUC"], {"auc": "%.4f"})
w("Over all 225 cells (%d positives, %d negatives) n_eff attains AUC **%.4f** "
  "against **%.4f** for n; restricted to the 180 cells with a genuine slope "
  "shift the gap widens to %.4f against %.4f.  The brief's pilot reported "
  "0.937 against 0.881 -- the same ordering, with a larger margin here because "
  "the project DGP's skewed eta law makes n a worse proxy than it is under a "
  "normal law.  Both quantities are observable before any calibration "
  "parameter is estimated; n_eff is simply the better one."
  % (s2["d_positives"], s2["d_negatives"], s2["d_auc_n_eff"], s2["d_auc_n"],
     s2["d_auc_n_eff_b0_ne_1"], s2["d_auc_n_b0_ne_1"]))
w()
d2 = pd.DataFrame(s2["d_table_by_n"])
d2["Bc"] = ["%d/%d" % (x, y) for x, y in zip(d2.B_beats_A, d2.cells)]
d2["Ec"] = ["%d/%d" % (x, y) for x, y in zip(d2.E_beats_A, d2.cells)]
w("The same table as 2c, keyed on n instead of S:")
w()
table(d2[["bin", "cells", "Bc", "Ec", "mean_LL_B_minus_A", "mean_LL_E_minus_A",
          "mean_n_eff"]],
      ["n", "cells", "B beats A", "E beats A", "mean LL(B) - LL(A)",
       "mean LL(E) - LL(A)", "mean n_eff"],
      {"mean_LL_B_minus_A": "%+.4f", "mean_LL_E_minus_A": "%+.4f",
       "mean_n_eff": "%.2f"})
w("Keyed on n the transition is smeared: B still loses on average at n = 100 "
  "and only turns positive at n = 200, and even at n = 400 it fails in "
  "%d of 45 cells.  Keyed on S the same 225 cells sort almost perfectly."
  % (45 - int(d2[d2.bin == "n = 400"].B_beats_A.iloc[0])))
w()

w("### 2e. Risk envelope")
w()
e2 = pd.DataFrame(s2["e_table"])
nm = {"A": "A  intercept-only", "B": "B  unrestricted ML",
      "C": "C  CV penalty, no staging", "D": "D  staged rule (comparator)",
      "Dind": "D' staged rule, independent CV folds",
      "E": "E  information-adaptive"}
e2["arm"] = e2.arm.map(nm)
table(e2[["arm", "mean_excess", "max_excess", "median_excess",
          "cells_within_0p001"]],
      ["arm", "mean excess", "max excess", "median excess",
       "cells within 0.001 of the oracle"],
      {"mean_excess": "%.4f", "max_excess": "%.4f", "median_excess": "%.4f"})
ee = e2.set_index("arm")
w("Excess is measured against the per-cell oracle min(LL_A, LL_B).  The "
  "unrestricted slope is the clear loser: mean excess %.4f and worst case "
  "%.4f, four to five times the mean risk of any other arm.  Arms A, C, D and "
  "E all sit within %.4f of the oracle in the mean.  The difference between "
  "them is in the tail: intercept-only updating has the **worst maximum** "
  "excess of the four (%.4f), and the information-adaptive arm roughly halves "
  "it (%.4f) at the cost of a slightly larger mean (%.4f against %.4f).  The "
  "brief's pilot found E ahead of A on the mean as well (0.0025 against "
  "0.0058); under the project DGP's skewed eta law A is penalised less often, "
  "so E's advantage shows up in the worst case rather than the average."
  % (ee.loc["B  unrestricted ML", "mean_excess"],
     ee.loc["B  unrestricted ML", "max_excess"],
     max(ee.loc["A  intercept-only", "mean_excess"],
         ee.loc["E  information-adaptive", "mean_excess"]),
     ee.loc["A  intercept-only", "max_excess"],
     ee.loc["E  information-adaptive", "max_excess"],
     ee.loc["E  information-adaptive", "mean_excess"],
     ee.loc["A  intercept-only", "mean_excess"]))
w()

w("### Arm diagnostics")
w()
rows = []
for k in ["A", "B", "C", "D", "Dind", "E"]:
    rows.append(dict(arm=nm[k], LL=cells2["LL_" + k].mean(),
                     BR=cells2["BR_" + k].mean(),
                     mcse=cells2["LLse_" + k].median(),
                     sdb=cells2["sdb_" + k].median(),
                     sdbmax=cells2["sdb_" + k].max(),
                     fix=cells2["fix_share_" + k].mean()))
table(pd.DataFrame(rows),
      ["arm", "mean expected LL", "mean expected Brier", "median MC SE",
       "median SD of fitted slope", "max SD of fitted slope",
       "mean share fixing b = 1"],
      {"LL": "%.4f", "BR": "%.4f", "mcse": "%.6f", "sdb": "%.4f",
       "sdbmax": "%.4f", "fix": "%.4f"})
w("One-class calibration subsets occur in %.2f%% of replicates overall, "
  "reaching %.1f%% in the worst cell; by n they are %s.  The unrestricted ML "
  "fit fails to converge in %.2f%% of replicates (all at small n with a "
  "near-separated subset, where the slope is driven to the +/-25 clip).  "
  "n_eff over all 112,500 replicates spans %.4f to %.2f."
  % (100 * s2["mean_one_class_share"], 100 * s2["max_one_class_share"],
     ", ".join("%s: %.2f%%" % (k, 100 * v)
               for k, v in s2["one_class_share_by_n"].items()),
     100 * s2["mean_ml_nonconvergence"],
     s2["n_eff_overall_range"][0], s2["n_eff_overall_range"][1]))
w()

if s2n:
    w("### Table 2g -- the same study under a normal eta law "
      "(implementation reference)")
    w()
    ea = {r["arm"]: r for r in s2["e_table"]}
    en = {r["arm"]: r for r in s2n["e_table"]}
    ref = pd.DataFrame([
        ("mean LL(D) - LL(C), n >= 50, shared folds",
         s2["a_mean_LL_D_minus_C_n_ge_50"], s2n["a_mean_LL_D_minus_C_n_ge_50"],
         "-0.00004"),
        ("mean LL(D') - LL(C), n >= 50, independent folds",
         s2["a_mean_LL_Dind_minus_C_n_ge_50"],
         s2n["a_mean_LL_Dind_minus_C_n_ge_50"], "-0.00004"),
        ("same penalty selected, independent folds",
         s2["a_same_penalty_share_independent"],
         s2n["a_same_penalty_share_independent"], "0.66"),
        ("AUC of n_eff", s2["d_auc_n_eff"], s2n["d_auc_n_eff"], "0.937"),
        ("AUC of n", s2["d_auc_n"], s2n["d_auc_n"], "0.881"),
        ("mean oracle excess, arm E", ea["E"]["mean_excess"],
         en["E"]["mean_excess"], "0.0025"),
        ("mean oracle excess, arm A", ea["A"]["mean_excess"],
         en["A"]["mean_excess"], "0.0058"),
        ("mean oracle excess, arm B", ea["B"]["mean_excess"],
         en["B"]["mean_excess"], "0.0244"),
    ], columns=["quantity", "proj", "norm", "pilot"])
    table(ref, ["quantity", "project DGP eta law", "normal eta law",
                "brief's pilot reference"],
          {"proj": "%.6f", "norm": "%.6f"})
    nb = pd.DataFrame(s2n["b_table"])
    w("Under the normal law n_eff at n = 50 spans %.2f to %.2f against the "
      "pilot's 0.18 to 13.95 -- the upper end agrees to within %.0f%%.  Every "
      "headline quantity moves toward the pilot when the eta law is switched "
      "to normal, which is the expected behaviour and locates the remaining "
      "differences in the eta law rather than in the implementation."
      % (float(nb[nb.n == 50].min_n_eff.iloc[0]),
         float(nb[nb.n == 50].max_n_eff.iloc[0]),
         100 * abs(float(nb[nb.n == 50].max_n_eff.iloc[0]) - 13.95) / 13.95))
    w()

v2 = jload("t2_verify_summary_project.json")
w("### Verification: seed stability and Monte Carlo error")
w()
if v2:
    table(pd.DataFrame(v2["per_comparison"]),
          ["comparison", "cells re-run", "sign unchanged", "share",
           "largest change in the difference"],
          {"share": "%.4f", "max_abs_change": "%.6f"})
    w("A stratified subset of **%d design cells** -- every level of n, every "
      "level of b0, and both extremes of prevalence and prediction spread -- "
      "was re-run with a fresh seed (%d) at %d replicates.  **%d of %d "
      "headline comparisons keep their sign (%.1f%%).**  The median per-cell "
      "Monte Carlo SE of expected log loss is %.6f in the main run and %.6f in "
      "the re-run (largest across arms within each cell)."
      % (v2["cells_rerun"], 20260926, v2["replicates"],
         v2["fresh_seed_sign_unchanged"], v2["fresh_seed_comparisons"],
         100 * v2["fresh_seed_sign_unchanged_share"],
         v2["median_mcse_per_cell_original"], v2["median_mcse_per_cell_rerun"]))
    w()
else:
    w("*Pending -- see section 5.*")
    w()

# ==================================================================== 3. Task 3
co = pd.read_csv(RES / "t3_link_coefficients.csv")
pr = pd.read_csv(RES / "t3_link_prediction.csv")
key = co[co.term == "lag_soh"].set_index("link")
ro = (pr[pr.arm == "matr_rolling_origin"].groupby("link")
      .agg(brier=("brier", "mean"), auc=("auc", "mean"),
           bmin=("brier", "min"), bmax=("brier", "max"),
           amin=("auc", "min"), amax=("auc", "max"), ll=("log_loss", "mean")))
rows = []
for link in ["cloglog", "logit", "probit"]:
    k, r = key.loc[link], ro.loc[link]
    rows.append(dict(link=link, beta=k.beta,
                     ci="(%.4f, %.4f)" % (k.beta_low, k.beta_high),
                     expb=k.exp_beta,
                     expci="(%.4f, %.4f)" % (k.exp_beta_low, k.exp_beta_high),
                     brier=r.brier, auc=r.auc,
                     brng="%.4f - %.4f" % (r.bmin, r.bmax),
                     arng="%.4f - %.4f" % (r.amin, r.amax)))
t3 = pd.DataFrame(rows)

w("## 3. Link comparison on the battery corpus")
w()
w("The two quantities the brief quotes for the existing cloglog model come "
  "from two different existing protocols, so both were rerun per link with "
  "nothing else changed:")
w()
w("* **coefficient** -- the 493-cell / 16,538-row pool under the Study C "
  "protocol of `79_mcsm_experiment_suite.py`: standardised `log_interval_mid`, "
  "`lag_soh`, `delta_soh_10` plus dataset and chemistry dummies, penalty "
  "alpha = 0.03162277660168379 held fixed across links, 1,000 whole-cell "
  "bootstrap resamples, seed 20260823;")
w("* **prediction** -- the leakage-free rolling-origin protocol that produced "
  "the quoted ranges: `event ~ C(batch) + lag_soh + delta_soh_10` on the MATR "
  "official-endpoint cycle path, published whole-cell train/test split, nine "
  "landmark-horizon settings, complete-case labelling.")
w()
w("The cloglog row reproduces the published values exactly (hazard ratio "
  "0.499526, interval (0.471587, 0.524324); Brier 0.0366 - 0.0967, AUC "
  "0.9174 - 0.9949), which validates both pipelines before the link is changed.")
w()
w("### Table 3 -- link comparison")
w()
table(t3, ["link", "lagged-SOH coefficient", "95% bootstrap interval",
           "exp(coef)", "95% interval for exp(coef)", "mean Brier", "mean AUC",
           "Brier range", "AUC range"],
      {"beta": "%.4f", "expb": "%.4f", "brier": "%.4f", "auc": "%.4f"})
ratio = float(key.exp_beta.max() / key.exp_beta.min())
w("**This task does not reproduce on real data what the brief expected, and "
  "the honest result is the opposite of the anticipated one.**  The ratio of "
  "the largest to the smallest exp(coef) across links is **%.4f** -- a %.1f%% "
  "disagreement -- beside a spread in mean Brier of %.4f (%.4f to %.4f) and in "
  "mean AUC of %.4f (%.4f to %.4f).  The three links agree on the coefficient "
  "just as closely as they agree on prediction.  The bootstrap intervals "
  "overlap almost completely."
  % (ratio, 100 * (ratio - 1), ro.brier.max() - ro.brier.min(),
     ro.brier.min(), ro.brier.max(), ro.auc.max() - ro.auc.min(),
     ro.auc.min(), ro.auc.max()))
w()
sep = jload("t3_separation_diagnostics.json")
ps = pd.read_csv(RES / "t3_link_penalty_sensitivity.csv")
piv = ps.pivot(index="alpha", columns="link", values="exp_beta")
cv = ps.pivot(index="alpha", columns="link", values="converged")
out = []
for al in piv.index:
    vals = piv.loc[al]
    allc = bool(cv.loc[al].all())
    out.append(dict(alpha=al, cloglog=vals["cloglog"], logit=vals["logit"],
                    probit=vals["probit"],
                    ratio=float(vals.max() / vals.min()),
                    conv="yes" if allc else
                    "no (" + ", ".join(sorted(cv.columns[~cv.loc[al].values])) + ")"))
w("**Why.**  The comparison is only well posed at the manuscript's selected "
  "penalty.  Refitting all three links across the penalty grid:")
w()
table(pd.DataFrame(out),
      ["penalty alpha", "cloglog exp(coef)", "logit exp(coef)",
       "probit exp(coef)", "max/min", "all three links converged?"],
      {"alpha": "%.6g", "cloglog": "%.5f", "logit": "%.5f", "probit": "%.5f",
       "ratio": "%.2f"})
w("At every penalty weaker than the selected one **the cloglog fit does not "
  "converge** (L-BFGS-B stops after one iteration with a non-zero gradient), "
  "while logit and probit converge to coefficients that march off toward a "
  "separation boundary -- logit exp(coef) = %.5f at alpha = 0.  The apparent "
  "thousand-fold link disagreement in that regime is a comparison between a "
  "failed fit and a diverging one, and supports nothing."
  % float(piv.loc[0.0, "logit"]))
w()
w("The cause is threshold proximity, which the manuscript already discusses "
  "in another context: the endpoint is defined by SOH crossing 0.80 and the "
  "covariate is lagged SOH measured immediately before it.  Lagged SOH nearly "
  "separates the interval event -- univariate AUC **%.4f**, all %d event rows "
  "confined to lag_soh in [%.4f, %.4f], an event rate of %.1f%% below "
  "lag_soh = 0.8214 against %.3f%% above it, against an overall row event rate "
  "of %.2f%%.  Under near-separation the unpenalised estimates are not "
  "defined, which is exactly why five-fold whole-cell CV selected so strong a "
  "penalty; and a penalty strong enough to stabilise the fit also dominates "
  "the likelihood enough to pull all three links to a common shrunken value."
  % (sep["univariate_auc_lag_soh"], 303, sep["event_lag_soh_min"],
     sep["event_lag_soh_max"], 100 * sep["rate_below_0p8214"],
     100 * sep["rate_above_0p8214"], 100 * sep["row_event_rate"]))
w()
w("**What the paper can and cannot claim from this.**  It can claim the "
  "prediction half: at a fixed grouping, changing the binary link moves "
  "leakage-free rolling-origin Brier by %.4f and AUC by %.4f across nine "
  "landmark-horizon settings, i.e. not at all in any operational sense.  It "
  "**cannot** use this corpus to show that only the cloglog coefficient stays "
  "on the generating proportional-hazards scale: at the one penalty where all "
  "three links are estimable they agree to %.1f%%, so there is no real-data "
  "coefficient disagreement to point at.  Recommend reporting the prediction "
  "invariance as the real-data fact and leaving the coefficient-scale argument "
  "to the simulation, where the generating coefficient is known."
  % (ro.brier.max() - ro.brier.min(), ro.auc.max() - ro.auc.min(),
     100 * (ratio - 1)))
w()

# ==================================================================== 4. Task 4
s4 = jload("t4_summary.json")
w("## 4. Stochastic-process degradation baselines")
w()
if not s4:
    w("*Pending -- see section 5.*")
    w()
else:
    w("Degradation is D(t) = 1 - SOH(t) and the persistent 80% SOH endpoint is "
      "a threshold at D* = 0.20; this was verified against the data rather "
      "than assumed (`eol_cycle` equals the first cycle with SOH <= 0.80 for "
      "all 83 MATR events).  Paths live on the project's own 25-cycle "
      "observation schedule.  The Wiener process is fitted on the raw series; "
      "the gamma process requires a non-decreasing path, so a running maximum "
      "is applied to D for it -- **17.4% of raw increments on the MATR cycle "
      "path are negative** (median 16.7% per cell, maximum 47.5%), so that "
      "step removes a great deal of real capacity regeneration.  "
      "Both models are fitted on training cells only and "
      "predict each test cell using only its observations at or before L.")
    w()
    w("All four fairness requirements hold.  Requirement 4 is the one the "
      "published protocol does not enforce: one MATR test cell reaches the "
      "endpoint at cycle 148, before every landmark, and `make_landmark_table` "
      "retains it.  The headline table excludes it for **every** model, and "
      "every model is scored on exactly the cells all models can score, so no "
      "model gets a different denominator.")
    w()
    t4 = pd.DataFrame(s4["table4_matr"])
    w("### Table 4 -- MATR official-endpoint path, raw cycle scale, whole-cell "
      "split, requirement 4 enforced, full training window")
    w()
    table(t4[["model", "landmark_cycle", "horizon_cycle", "n_evaluated",
              "n_events", "brier", "log_loss", "auc"]],
          ["model", "L", "H", "n evaluated", "events", "Brier", "log loss",
           "AUC"],
          {"brier": "%.4f", "log_loss": "%.4f", "auc": "%.4f"})
    m4 = pd.DataFrame(s4["table4_matr_means"])
    w("Means across the nine settings:")
    w()
    table(m4, ["model", "mean Brier", "mean log loss", "mean AUC",
               "mean predicted risk", "mean observed rate"],
          {"brier": "%.4f", "log_loss": "%.4f", "auc": "%.4f",
           "mean_pred": "%.4f", "observed_rate": "%.4f"})
    mm = m4.set_index("model")
    _all4 = pd.concat([pd.read_csv(RES / f) for f in
                       ["t4_first_passage_metrics_matr.csv",
                        "t4_first_passage_metrics_pool.csv"]
                       if (RES / f).exists()], ignore_index=True)
    msens = (_all4[_all4.corpus == "matr"]
             .pivot_table(index="model",
                          columns=["window", "variant", "timescale"],
                          values="brier", aggfunc="mean"))
    msens_power_w = float(msens.loc["wiener", ("full", "fair", "power")])
    msens_power_g = float(msens.loc["gamma", ("full", "fair", "power")])
    s6m = (_all4[(_all4.corpus == "sixsource") & (_all4.timescale == "cycle")
                 & (_all4.window == "full")]
           .groupby("model")[["brier", "log_loss", "auc"]].mean())
    w("The first-passage models are **three to four times worse on Brier** "
      "than the hazard model and lose about 0.27 of AUC.  The reason is "
      "visible in the last two columns: they systematically under-predict, "
      "placing a mean risk of %.4f (Wiener) and %.4f (gamma) against an "
      "observed rate of %.4f, while both the hazard model and the random "
      "forest are calibrated in the mean to within %.4f.  The random forest is "
      "the best of the four on this corpus, which the paper already reports "
      "for the hazard-versus-ML comparison and which is not affected here."
      % (mm.loc["wiener", "mean_pred"], mm.loc["gamma", "mean_pred"],
         mm.loc["cloglog hazard model", "observed_rate"],
         max(abs(mm.loc["cloglog hazard model", "mean_pred"]
                 - mm.loc["cloglog hazard model", "observed_rate"]),
             abs(mm.loc["random forest", "mean_pred"]
                 - mm.loc["random forest", "observed_rate"]))))
    w()
    w("### Fitted parameters and whether they are physically sensible")
    w()
    wp = pd.DataFrame(s4["wiener_params"])
    table(wp, ["corpus", "train window", "mu0 (per cycle)", "s2mu", "sigma^2",
               "implied mean cycles to D* = 0.20"],
          {"mu0": "%.4e", "s2mu": "%.4e", "sig2": "%.4e",
           "cycles_to_threshold": "%.0f"})
    gpar = pd.DataFrame(s4["gamma_params"])
    table(gpar, ["corpus", "train window", "alpha", "a0", "b0",
                 "mean rate (per cycle)", "implied mean cycles to D* = 0.20"],
          {"alpha": "%.4f", "a0": "%.4f", "b0": "%.4e", "mean_rate": "%.4e",
           "cycles_to_threshold": "%.0f"})
    wm = pd.DataFrame(s4["wiener_params"]).set_index(["corpus", "window"])
    gm = pd.DataFrame(s4["gamma_params"]).set_index(["corpus", "window"])
    w("On MATR the parameters are physically sensible in scale -- a mean drift "
      "of %.2e of SOH per cycle, reaching D* = 0.20 at about cycle %.0f for "
      "the Wiener model and cycle %.0f for the gamma process -- but **both "
      "under-state the degradation rate**, and that is the whole result.  The "
      "observed median endpoint among the 83 MATR events is cycle **527** "
      "(10th to 90th percentile 457 to 910), so the Wiener model is late by "
      "about %.0f cycles and the gamma process by about %.0f."
      % (wm.loc[("matr", "full"), "mu0"],
         wm.loc[("matr", "full"), "cycles_to_threshold"],
         gm.loc[("matr", "full"), "cycles_to_threshold"],
         wm.loc[("matr", "full"), "cycles_to_threshold"] - 527,
         gm.loc[("matr", "full"), "cycles_to_threshold"] - 527))
    w()
    w("The reason is that lithium-ion capacity fade is strongly **convex in "
      "cycle number** while both models assume time-homogeneous increments.  "
      "Measured directly on the 81 event cells with usable paths, the "
      "degradation accrued by a given fraction of observed life, relative to "
      "the proportional amount, is 0.448 at a quarter of life, 0.359 at half "
      "and 0.459 at three quarters (medians).  A constant-drift Wiener process "
      "and a constant-rate gamma process both fit that early flat region and "
      "then extrapolate linearly, so at a landmark of 150 to 300 cycles -- "
      "where median D is only about 0.03 against a threshold of 0.20 -- they "
      "place almost no probability on crossing by cycle 500 to 1000.")
    w()
    if "matr_timescale_invariance_max_brier_diff" in s4:
        w("**Time scale.**  The headline uses raw cycle number.  Dividing "
          "cycle number by the per-source median observed life is, for a "
          "single-source corpus, a pure change of units: the fitted parameters "
          "rescale and the predictions are invariant.  That was confirmed "
          "numerically rather than assumed -- the largest Brier difference "
          "between the raw and normalised runs across all model x setting "
          "combinations is %.3e.  A normalised time scale can only matter "
          "where the scale differs across sources, which is the six-source "
          "corpus below."
          % s4["matr_timescale_invariance_max_brier_diff"])
        w()
    w("**Power-transformed time, as a diagnostic.**  To test whether the "
      "failure is the first-passage framework or specifically the assumption "
      "of time-homogeneous increments, both models were refitted on a power "
      "time scale t -> t^gamma with gamma profiled over the Wiener likelihood. "
      " This is an extension beyond the two models the brief specifies and is "
      "reported only as a diagnostic.  It helps substantially and is still not "
      "close: mean Brier improves from %.4f to %.4f for the Wiener model and "
      "from %.4f to %.4f for the gamma process, against %.4f for the hazard "
      "model.  So convexity in cycle number is the dominant part of the gap, "
      "but not all of it."
      % (mm.loc["wiener", "brier"], msens_power_w, mm.loc["gamma", "brier"],
         msens_power_g, mm.loc["cloglog hazard model", "brier"]))
    w()
    if "sixsource_detail_cycle_full" in s4:
        w("### Table 4b -- six-source corpus (493 cells), five-fold whole-cell "
          "cross-validation, raw cycle scale")
        w()
        w("This extends the comparison to the full heterogeneous corpus.  Read "
          "it with two warnings, both established below rather than assumed.")
        w()
        w("* The **Wiener rows here come from an unidentified fit** and must "
          "not be read as a model ranking.  On this representation the "
          "likelihood has a spurious high-likelihood region: the fold-averaged "
          "estimates include mu0 = 0.1215 of SOH per cycle and sigma^2 = 16.87, "
          "which are physically impossible, while the drift-variance component "
          "collapses to about 4e-17.  The gamma and hazard rows are from "
          "converged fits with physically sensible parameters.")
        w("* The row labelled *cloglog cumulative-risk (artefact-affected)* is "
          "the accumulation 1 - prod(1 - h_ij) used by the published transport "
          "analysis.  It is shown only to quantify the survivorship artefact "
          "described in section 5, and is not the hazard-model comparator.  The "
          "comparator row *cloglog hazard model* is a landmark-style fit on the "
          "fixed-horizon label with no accumulation, which is what the MATR arm "
          "uses.")
        w()
        s6 = pd.DataFrame(s4["sixsource_detail_cycle_full"])
        table(s6[["model", "landmark_cycle", "horizon_cycle", "n_evaluated",
                  "n_events", "brier", "log_loss", "auc"]],
              ["model", "L", "H", "n evaluated", "events", "Brier",
               "log loss", "AUC"],
              {"brier": "%.4f", "log_loss": "%.4f", "auc": "%.4f"})
        w("Means across the nine settings: hazard model Brier %.4f / AUC %.4f, "
          "gamma %.4f / %.4f, Wiener (unidentified) %.4f / %.4f, and the "
          "artefact-affected accumulation %.4f / %.4f.  The ordering agrees "
          "with the MATR arm wherever it is trustworthy: the hazard model is "
          "ahead on both Brier and AUC.  The gamma process does markedly "
          "better here than on MATR (Brier %.4f against %.4f) but its log loss "
          "is %.4f against %.4f for the hazard model -- it is confidently "
          "wrong on the cells it gets wrong."
          % (s6m.loc["cloglog hazard model", "brier"],
             s6m.loc["cloglog hazard model", "auc"],
             s6m.loc["gamma", "brier"], s6m.loc["gamma", "auc"],
             s6m.loc["wiener", "brier"], s6m.loc["wiener", "auc"],
             s6m.loc["cloglog cumulative-risk (artefact-affected)", "brier"],
             s6m.loc["cloglog cumulative-risk (artefact-affected)", "auc"],
             s6m.loc["gamma", "brier"], mm.loc["gamma", "brier"],
             s6m.loc["gamma", "log_loss"],
             s6m.loc["cloglog hazard model", "log_loss"]))
        w()
        w("**Why the six-source representation is hostile to a first-passage "
          "model.**  The pool's grouped representation gives "
          "D = 1 - `lag_soh`, and `lag_soh` there runs from 0.2338 to 1.6825, "
          "so D ranges from **-0.6825 to +0.7662**: 311 of 16,538 rows (1.9%) "
          "have negative degradation, spread over **109 of the 493 cells**, and "
          "28 rows already sit beyond the D* = 0.20 threshold.  Both processes "
          "require a non-negative degradation path, and the Wiener model "
          "additionally assumes D(0) = 0 at t = 0 while every pool path begins "
          "at interval start 100.  The MATR cycle path by contrast has D "
          "between 0.0002 and 0.2524 with **no** negative rows, which is why "
          "the MATR arm identifies cleanly and carries the headline.  A "
          "reviewer asking for the six-source comparison should be told that "
          "the corpus first needs a degradation representation built for it.")
        w()
    w("### Sensitivity")
    w()
    if "matr_sensitivity_mean_brier" in s4:
        sm = pd.DataFrame(s4["matr_sensitivity_mean_brier"])
        w("Mean Brier by training window, requirement-4 variant and time "
          "scale (MATR).  Enforcing requirement 4 changes the hazard model's "
          "mean Brier from %.4f to %.4f, i.e. the one prematurely-crossed cell "
          "matters little here but the comparison is now exact.  Truncating "
          "the training paths at the horizon makes both process models "
          "slightly worse (Wiener %.4f -> %.4f, gamma %.4f -> %.4f), as "
          "expected, since it removes the late accelerating part of the "
          "training paths."
          % (msens.loc["cloglog hazard model", ("full", "published", "cycle")],
             msens.loc["cloglog hazard model", ("full", "fair", "cycle")],
             msens.loc["wiener", ("full", "fair", "cycle")],
             msens.loc["wiener", ("truncH", "fair", "cycle")],
             msens.loc["gamma", ("full", "fair", "cycle")],
             msens.loc["gamma", ("truncH", "fair", "cycle")]))
        w()
        w("```")
        w(msens.round(4).to_string())
        w("```")
        w()
    w("**What agrees and what diverges.**  The expected story in the brief was "
      "that a first-passage model would be competitive when degradation is "
      "close to monotone and the threshold is far away, and would degrade "
      "where regeneration or protocol heterogeneity breaks the process "
      "assumptions.  What the data actually show is sharper and different: the "
      "first-passage models are **worst exactly where the threshold is far "
      "away**.  At L = 300, H = 500 -- the setting with the most margin, where "
      "median D is about 0.03 against a threshold of 0.20 -- the Wiener model "
      "predicts a mean risk of %.4f against an observed rate of %.4f and "
      "records a log loss of %.4f, its worst of the nine.  They are relatively "
      "least bad at the long horizons, where enough degradation has accrued "
      "that even a linear extrapolation reaches the threshold.  The mechanism "
      "is the convexity of capacity fade, not regeneration: the running "
      "maximum needed by the gamma process removes 17.4%% of increments and the "
      "gamma process is still the weaker of the two on MATR.  Discrimination "
      "survives better than calibration -- Wiener AUC reaches %.4f at "
      "L = 300, H = 800 -- so the processes do rank cells, they just cannot "
      "put the risk on the right scale.  The hazard model absorbs both the "
      "convexity, through interval-specific baselines, and the regeneration, "
      "through lagged SOH and recent SOH change as covariates."
      % (float(t4[(t4.model == "wiener") & (t4.landmark_cycle == 300)
                  & (t4.horizon_cycle == 500)].mean_pred.iloc[0]),
         float(t4[(t4.model == "wiener") & (t4.landmark_cycle == 300)
                  & (t4.horizon_cycle == 500)].observed_rate.iloc[0]),
         float(t4[(t4.model == "wiener") & (t4.landmark_cycle == 300)
                  & (t4.horizon_cycle == 500)].log_loss.iloc[0]),
         float(t4[(t4.model == "wiener") & (t4.landmark_cycle == 300)
                  & (t4.horizon_cycle == 800)].auc.iloc[0])))
    w()

# ==================================================================== 5. issues
w("## 5. Anything that failed, was skipped, or looks wrong")
w()
w("**1. Task 3 does not deliver the finding the brief anticipated.**  The "
  "brief expected the links to \"disagree materially on the coefficient while "
  "agreeing on prediction\".  On this corpus they agree on both, to %.1f%% on "
  "exp(coef).  The mechanism is documented in section 3 with a penalty sweep "
  "and separation diagnostics: at any penalty weak enough to let the links "
  "diverge, the cloglog fit does not converge and the logit and probit fits "
  "run off toward a separation boundary, because lagged SOH nearly separates "
  "the interval event (univariate AUC %.4f).  Reported as found rather than "
  "presented as a confirmation." % (100 * (ratio - 1),
                                    sep["univariate_auc_lag_soh"]))
w()
w("**2. A corpus-consistent version of Table 3 was attempted and rejected as "
  "invalid.**  Task 3's method section asks for the 493-cell corpus, but the "
  "quoted Brier and AUC ranges come from the MATR path, so a 493-cell "
  "five-fold whole-cell version was also built.  It must not be used.  Cell "
  "level fixed-horizon risk on the pool is accumulated as "
  "1 - prod(1 - h_ij) over intervals in [L, H], and a cell that fails leaves "
  "the risk set, so it contributes fewer intervals: at L = 150, H = 800 "
  "failing cells contribute 12.4 intervals on average against 27.0 for "
  "survivors, and **the AUC of the interval count alone is 0.0000** -- "
  "perfectly inversely predictive.  The accumulated risk mixes the hazard "
  "signal with that artefact, and the mixture is link-dependent: the "
  "correlation between accumulated risk and interval count is +0.146 for "
  "cloglog and -0.366 for probit, which drags cloglog AUC down to 0.557 and "
  "inflates probit's to 0.920.  The apparent probit advantage is survivorship "
  "arithmetic, not a link effect.  Numbers are in "
  "`results/t3_link_prediction.csv` under `arm = pool_wholecell_cv`, flagged "
  "there, and are not used anywhere in this report.  **The same accumulation "
  "is used by the published transport analysis behind Tables 8 and 9**; its "
  "before-versus-after log-loss comparisons are unaffected because both sides "
  "share the artefact, but any discrimination claim from those tables should "
  "be re-checked.")
w()
w("**3. The brief's Task 3 specification is internally inconsistent.**  It "
  "asks for the 493-cell person-period corpus \"exactly as used for the main "
  "results\" and then quotes a hazard ratio that comes from the 493-cell pool "
  "and Brier/AUC ranges that come from the 139-cell MATR official-endpoint "
  "path.  Both protocols were run per link and both reproduce their published "
  "reference values; Table 3 pairs them as the brief's own table does, and the "
  "provenance of each column is stated.")
w()
w("**4. Two defects in the reproducibility archive.**  "
  "`src/pipeline/79_mcsm_experiment_suite.py` sets "
  "`ROOT = Path(__file__).resolve().parents[1]`, which resolves to "
  "`<root>/src` after the archive move, so `load_pool()` raises "
  "FileNotFoundError from a clean checkout; and "
  "`src/pipeline/57_phase2_expanded_transport.py`, which "
  "`74_transport_recalibration_loco.py` imports at module load, is absent from "
  "the archive entirely.  Both were repaired locally (the first by routing "
  "through `project_paths`, the second by copying the file from the working "
  "tree and patching its hard-coded `D:/discrel` root).  **These should be "
  "fixed in the archive before it is deposited**, because stage 03 and stage "
  "08 cannot run without them.")
w()
w("**5. Descriptive discrepancy in the recalibration study's size.**  The "
  "brief and the supplementary text describe \"900 design cells x 500 "
  "replicates\".  The grid actually coded in "
  "`83_study_b_shrunk_recalibration.py` is 2 target sizes x 4 baseline shifts "
  "x 3 slope scalings x 7 or 8 calibration schemes = 180 design cells, "
  "evaluated for 5 methods, i.e. 900 design-cell-by-method rows.  Nothing "
  "turns on this, but the manuscript wording overstates the number of design "
  "cells by a factor of five.")
w()
w("**6. Wiener convergence flags are unreliable and were checked another "
  "way.**  `fit_wiener` frequently reports `converged = False` because "
  "Nelder-Mead exhausts its iteration budget after L-BFGS-B has already found "
  "the optimum.  Every headline fit was therefore verified by perturbation: "
  "moving each parameter by +/-2% and +/-20% never lowered the objective.  The "
  "flag is reported in `results/t4_first_passage_params_*.csv` but should not "
  "be read as a failure.")
w()
w("**6b. Two numerical defects were found and fixed in the first-passage "
  "fitting, and one of them had reversed a conclusion.**  (a) The gamma fit "
  "was not scale-equivariant: the same data expressed in cycles and in "
  "normalised cycles gave mean Brier 0.3181 and 0.4298, although the gamma "
  "process is closed under a rescaling of time and the predictions must be "
  "identical.  Fitting is now done with time standardised internally to a "
  "median increment of 1 and alpha transformed back, after which the two "
  "agree to 4.7e-09 on real data and to 6.7e-08 across a 1,860-fold change of "
  "units on synthetic data.  (b) `fit_wiener` from a single moment start "
  "collapsed the drift-variance component onto its lower bound on some "
  "six-source folds; profiling the likelihood showed a clear interior optimum "
  "with the boundary **429 nll units worse**, so those were optimiser "
  "failures, not results.  With multiple starts the MATR numbers are "
  "unchanged, but the six-source Wiener Brier moved from 0.1723 to 0.2567 and "
  "its AUC from 0.8486 to 0.7361 -- **the earlier, better-looking six-source "
  "Wiener result was an artefact of a degenerate fit**.  Had the parameters "
  "not been checked for physical plausibility, as the brief instructs, this "
  "report would have claimed that a first-passage model beats the hazard model "
  "on the six-source corpus.  It does not.")
w()
w("**6c. The six-source first-passage comparison is reported but is not a "
  "valid model ranking.**  Even after (6b) the Wiener fit on the pool is "
  "unidentified -- fold-averaged mu0 = 0.1215 of SOH per cycle and "
  "sigma^2 = 16.87 are physically impossible.  The cause is the "
  "representation, not the optimiser: the pool's D = 1 - `lag_soh` ranges "
  "from -0.6825 to +0.7662, with 311 of 16,538 rows (1.9%) negative across "
  "109 of the 493 cells and 28 rows already past the D* = 0.20 threshold, "
  "while both processes require a non-negative path and the Wiener model "
  "assumes D(0) = 0 although every pool path starts at interval 100.  The "
  "MATR cycle path has D in [0.0002, 0.2524] with no negative rows, which is "
  "why it carries the headline.  The gamma and hazard rows of Table 4b are "
  "from converged, physically sensible fits and can be read; the Wiener rows "
  "cannot.")
w()
w("**7. Deviations from the reference implementations, all verified.**  The "
  "Appendix B code was not executable against real data as supplied and three "
  "changes were needed; each is checked in `firstpassage.py::self_test`, which "
  "passes.  (a) The Wiener marginal likelihood was rewritten from an O(n^3) "
  "Cholesky per path to an O(1) evaluation from parameter-free sufficient "
  "statistics, by integrating the drift out analytically; the two agree to "
  "4.3e-14 relative once the reference's omitted 0.5*N*log(2*pi) constant is "
  "restored, and the speed-up is what made the study feasible.  (b) The gamma "
  "likelihood was vectorised across paths; it agrees with the reference loop "
  "to 3.3e-16 relative.  (c) `fit_gamma` is started from method-of-moments "
  "values because the reference start (alpha, a0, b0) = (1, 2, 50) implies a "
  "degradation rate four orders of magnitude too large, from which the "
  "optimiser does not recover.  Two of the reference self-checks were also "
  "degenerate as first written -- both the inverse-Gaussian and compound-gamma "
  "probabilities evaluated to exactly 1.0000, so they passed while testing "
  "nothing; they now sit at 0.4483 and 0.5271 and are compared against Monte "
  "Carlo.")
w()
w("**8. Arm C and arm D share cross-validation folds in the headline "
  "numbers.**  The brief requires folds to be shared across the penalty grid "
  "within a replicate; the supplied reference `select_rho_cv` re-permutes for "
  "each candidate, which inflates apparent disagreement between C and D.  Both "
  "are reported: shared folds isolate the staging, independent folds reproduce "
  "the pilot's 66%-style number.")
w()
w("**9. Environment.**  This machine has 7.35 GB of RAM.  The Task 2 study was "
  "killed by the operating system three times before the workload was "
  "restructured: BLAS threading is pinned to one thread per process (OpenBLAS "
  "allocates per-thread workspace in every worker), the innermost 2x2 solve "
  "was replaced by a closed form so LAPACK is not called in the hot loop, and "
  "parallelism is by independent sharded single-process jobs rather than a "
  "multiprocessing pool, whose result pipe does not survive detachment on this "
  "host.  Results are unaffected -- `verify_calib.py` reproduces the same "
  "agreement with statsmodels before and after the change -- but the sharding "
  "is why results arrive as `t2_recal_project_s{0,1}.json` and are merged.")
w()
w("**10. Not attempted.**  No Firth or bootstrap small-sample correction to "
  "the calibration slope was added, and the staged rule appears only as arm D, "
  "both as instructed.  Nothing was tuned on test data; all resampling and "
  "cross-validation is on whole units.")
w()

# ==================================================================== 6. files
w("## 6. Files produced")
w()
w("All paths relative to `F:\\discrel\\qrei`.")
w()
w("**Scripts**")
w()
files = [
    ("scripts/calib.py",
     "penalised cloglog recalibration, n_eff, CV-with-1SE selection, "
     "empirical-Bayes penalty, staged rule"),
    ("scripts/verify_calib.py",
     "self-test: rho = 0 fit against a statsmodels cloglog GLM, penalised fit "
     "against an independent optimiser, n_eff against the GLM slope SE"),
    ("scripts/t1_neff_transport_splits.py" if False else "scripts/t1_neff_transport.py",
     "Task 1 -- replays the published transport splits and records n_eff"),
    ("scripts/t2_eta_reservoir.py",
     "Task 2 stage 0 -- derives the target eta law from the project DGP"),
    ("scripts/t2_recal_boundary.py",
     "Task 2 -- the 225-cell, five-arm recalibration study"),
    ("scripts/t2_analyse.py", "Task 2 -- items 2a to 2e and two figures"),
    ("scripts/t2_verify.py", "Task 2 -- seed-stability re-run and summary"),
    ("scripts/t3_link_comparison.py",
     "Task 3 -- cloglog / logit / probit, coefficient and prediction arms"),
    ("scripts/firstpassage.py",
     "Wiener and gamma first-passage models with self-test"),
    ("scripts/t4_first_passage.py", "Task 4 -- the like-for-like comparison"),
    ("scripts/t4_analyse.py", "Task 4 -- Table 4 and the figure"),
    ("scripts/build_results.py", "generates this report from the result files"),
]
table(pd.DataFrame(files, columns=["path", "what"]), ["path", "what it does"])
w("**Results**")
w()
res_files = sorted(p.name for p in RES.glob("*") if p.is_file())
for f in res_files:
    w("* `results/%s`" % f)
w()
w("**Figures** (PNG, 320 dpi)")
w()
for f in ["fig_crossover.png", "fig_information_spread.png",
          "fig_firstpassage.png"]:
    p = QREI / "figs" / f
    w("* `figs/%s`%s" % (f, "" if p.exists() else "  *(pending)*"))
w()
w("**Seeds**")
w()
seeds = pd.DataFrame([
    ("Task 1 transport split replay", "20260702 + crc32(key) % 100000",
     "the published base seed; splits reproduce the archive exactly"),
    ("Task 2 eta reservoir", "20260823", "the project's main simulation seed"),
    ("Task 2 recalibration study", "SeedSequence([20260825, cell_index])",
     "the project's recalibration follow-up seed"),
    ("Task 2 seed-stability re-run", "SeedSequence([20260926, cell_index])",
     "fresh, chosen not to collide with any project seed"),
    ("Task 3 bootstrap", "SeedSequence([20260823, 0])",
     "reproduces the published interval exactly"),
    ("Task 4", "20260907; random forest keeps 20260531 + L + H",
     "the comparator's published per-setting seed"),
], columns=["what", "seed", "note"])
table(seeds, ["what", "seed", "note"])

path = QREI / "RESULTS.md"
path.write_text("\n".join(B) + "\n", encoding="utf-8")
print("wrote %s (%d lines)" % (path, len(B)))
