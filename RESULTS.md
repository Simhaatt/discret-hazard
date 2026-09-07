# Results

Experiments for the QREI resubmission of *Grouped discrete-time hazard modelling for degradation data*.  Numbers, tables and figures only; the manuscript is not rewritten here.

All work was run from `F:\discrel\qrei`.  Every script, seed and output path is listed in section 6.

## 0. Repository discovery

All five required artefacts were located.  The authoritative source is the reproducibility archive `C:\Users\hp\Documents\csda-grouped-hazard-repro`, copied to `F:\discrel\qrei\repro` and pointed at the data drive `F:\discrel\data\processed` (the archive resolves paths through `src/utilities/project_paths.py`, so no absolute path was edited).

| artefact | what was found | status |
|---|---|---|
| Simulation DGP | `src/pipeline/79_mcsm_experiment_suite.py::simulate_person_period`, with `baseline_alpha` (smooth increasing log-time baseline) and `true_health_effect` (analytic state effect outside the fitted spline basis). At interval width 25, `jmax = 1000/25 = 40`, i.e. the J = 40 intervals of Section S1.1. Sequential first-event generation, independent censoring, one time-varying (`ar`) and one unit-level (`static`) nuisance covariate. Main seed 20260823. | found |
| Recalibration study | `src/pipeline/83_study_b_shrunk_recalibration.py`, seed 20260825. Contains `RHO_GRID = [0.01, 0.1, 1, 10, inf]`, `MIN_TUNE_CELLS = 20` and `candidate_grid = RHO_GRID if len(y) >= 50 else [0.1, 1, 10, inf]` -- exactly the 20/50 staging and the two grids described in the brief. | found |
| Battery corpus | `data/processed/phase2_expanded_person_period_grouped_25.csv`: 16,538 person-period rows, 493 cells, 303 harmonised events, 190 censored, six sources with the stated counts (MATR 139, BatteryLife SNL 61, CALB 27, HUST 77, SDU 70, Tongji 119). Verified that `eol_cycle` equals the first cycle with SOH <= 0.80 for all 83 MATR events, confirming the persistent 80% SOH endpoint and the threshold D* = 0.20. | found |
| Fitted source models | `src/pipeline/build_manuscript_tables.py::fixed_horizon_transport`, which fits the pooled cloglog hazard via `74_transport_recalibration_loco.py::fit_cloglog` (statsmodels GLM, cloglog link, `fit_regularized(alpha=0.01, L1_wt=0)`) on all sources but the held-out one, accumulates each target cell's landmark-to-horizon risk, and recalibrates on a 30% target-cell subset over 100 splits from base seed 20260702. Its output `table_D_fixed_horizon_external_transport_recalibration.csv` is what manuscript Tables 8 and 9 aggregate. | found |
| Landmark-horizon grid | `LANDMARKS = [150, 200, 300]` x `HORIZONS = [500, 800, 1000]` = nine settings (`build_manuscript_tables.py`). | found |

Both published cloglog reference values quoted in the brief were reproduced exactly before any new work was done, which fixes the protocol for Tasks 1, 3 and 4:

* the standardised lagged-SOH hazard ratio **0.499526** with 95% whole-cell bootstrap interval **(0.471587, 0.524324)** at penalty alpha = 0.03162277660168379 -- reproduced to all six quoted digits (Task 3, Table 3);
* the leakage-free rolling-origin Brier range **0.0366 - 0.0967** and AUC range **0.9174 - 0.9949** -- reproduced to 4.8e-7 against archived `table06_rolling_origin_performance.csv` (Task 3, Table 3).

Two defects in the archive had to be repaired before it would run; both are recorded in section 5.

## 1. Effective calibration information for transport splits

`n_eff` was computed on the **same** labelled subsets the published analysis used, not on fresh ones: the replay reconstructs the generator `default_rng(20260702 + crc32("<holdout>:<group>:<L>:<H>") % 100000)` and consumes it over the recalibration fractions 0.05, 0.10, 0.20, 0.30 in the original order, recording the 30% splits.  As a check the replay also recomputes the published before/after log loss: it agrees with archived `table_D_*.csv` at all 90 rows to **exactly zero** difference, so the `n_eff` values below describe the very splits behind manuscript Tables 8 and 9.  `n_eff` is evaluated at the null recalibration (a = 0, b = 1) and uses only the source model's predictions -- no outcome labels.

Each row averages 100 target-cell splits.

### Table 1 -- calibration information per split, over the nine landmark-horizon settings

| split | held-out unit | settings | labelled cells | events in subset | sd(eta) | n_eff | n_eff range over the nine | 1/sqrt(n_eff) | one-class share |
|---|---|---|---|---|---|---|---|---|---|
| leave-one-chemistry-out | NMC | 9 | 34.0 | 17.35 | 0.7333 | 4.72 | 3.57 - 5.63 | 0.46 | 0.000 |
| leave-one-chemistry-out | NCA | 9 | 16.7 | 14.69 | 0.6396 | 2.18 | 1.22 - 3.44 | 0.68 | 0.571 |
| leave-one-chemistry-out | LFP | 9 | 61.4 | 11.14 | 0.3055 | 1.90 | 1.15 - 3.37 | 0.73 | 0.000 |
| leave-one-chemistry-out | NMC/NCA | 9 | 3.0 | 2.67 | 0.0759 | 0.01 | 0.01 - 0.01 | 10.67 | 0.759 |
| leave-one-source-out | Tongji | 9 | 19.7 | 15.25 | 0.7574 | 2.20 | 0.94 - 3.62 | 0.67 | 0.186 |
| leave-one-source-out | BatteryLife SNL | 9 | 15.0 | 6.75 | 0.6114 | 2.12 | 1.38 - 2.86 | 0.69 | 0.000 |
| leave-one-source-out | SDU | 9 | 18.0 | 11.41 | 0.5418 | 1.97 | 0.51 - 3.15 | 0.71 | 0.022 |
| leave-one-source-out | MATR | 9 | 32.9 | 11.15 | 0.2837 | 0.97 | 0.46 - 1.82 | 1.01 | 0.000 |
| leave-one-source-out | HUST | 9 | 24.0 | 0.11 | 0.1217 | 0.18 | 0.05 - 0.35 | 2.33 | 0.892 |
| leave-one-source-out | CALB | 9 | 6.7 | 0.94 | 0.2228 | 0.13 | 0.01 - 0.22 | 2.73 | 0.351 |

**Which splits are information-poor.**  Of the 90 (split, landmark, horizon) combinations, **83 of 90 fall below n_eff = 4** and **37 of 90 fall below n_eff = 1**; 24 fall below 0.25.  Only 7 combinations reach n_eff >= 4, and every one of them is the NMC split.  Nine of the ten splits have a nine-setting mean below 4.  Across all 90 rows n_eff spans **0.0059 to 5.63**, so the asymptotic standard error of an unpenalised calibration slope spans **0.42 to 13.04**.  A calibration slope is simply not estimable at most of these splits.

The two smallest sources behave as the brief anticipated.  CALB, with 27 cells, yields 6.7 labelled cells and n_eff = 0.13 (slope SE 2.7); the NMC/NCA chemistry, with 9 cells, yields 3 labelled cells and n_eff = 0.0088 (slope SE 10.7), and 76% of its calibration subsets contain only one outcome class.  These are findings, not defects.

**The more consequential finding is that labelled sample size does not order the information.**  Across the 90 rows the Spearman correlation between the number of labelled cells and n_eff is only **+0.468**, while the correlation with the spread of the source predictions, sd(eta), is **+0.900** (events in the subset: +0.739).  The product n * sd(eta)^2 correlates at +0.967, which is what the definition of n_eff implies.  The clearest inversion in the table: HUST supplies 24 labelled cells but n_eff = 0.18, whereas NCA supplies 17 labelled cells and n_eff = 2.18 -- **1.4 times fewer labelled units carrying 12 times more calibration information**.  HUST's subsets are large but almost eventless (0.11 events on average, 89% one-class) and its source predictions barely vary (sd(eta) = 0.122).  MATR is the same story more mildly: 33 labelled cells, 11.1 events, yet n_eff = 0.97 because sd(eta) is only 0.284.

This is the reviewer's objection made numerical on the paper's own data: a rule keyed on the number of labelled units would have treated HUST as better supported than NCA and been wrong by an order of magnitude.

Full detail, one row per (split, landmark, horizon), is in `results/t1_neff_transport_splits.csv`.

## 2. Recalibration boundary study

**Design.**  Labelled calibration units n in {25, 50, 100, 200, 400} x target event prevalence in {0.05, 0.20, 0.50} x spread of source predictions sd(eta) in {0.5, 1.0, 2.0} x true target calibration slope b0 in {0.7, 1.0, 1.25, 1.5, 2.0} = **225 design cells**, of which 180 have b0 != 1, with **500 replicates each (112,500 replicates)**.  Prevalence and prediction spread are set independently of n, which is what breaks the confound in the deleted design.  Seed 20260825, spawned per cell as `SeedSequence([20260825, cell_index])`.

**Target law from the project's own DGP.**  The brief asks for the project DGP rather than the normal linear predictor behind its pilot figures, so the law of eta was derived from Section S1.1 itself: a source cohort of 1,500 units was simulated with `simulate_person_period` at width 25 (J = 40 intervals), the project's penalised grouped cloglog hazard was fitted on it at the manuscript's selected penalty 0.0316228, an independent 20,000-unit target cohort was scored, and each unit's landmark-to-horizon risk was accumulated exactly as the battery analysis does, giving eta = log(-log(1 - r)) for 19,489 units (seed 20260823).  The source model is therefore genuinely misspecified for the target -- the true state effect is analytic and outside the fitted basis, and person-periods within a unit are serially dependent.

That eta law is **strongly non-normal**: skewness -1.88, excess kurtosis 4.03, Shapiro-Wilk decisively rejecting normality.  This matters, and it is the main reason some numbers below differ from the brief's pilot.  The law is represented as 2,000 equally weighted standardised quantile nodes, so expected held-out log loss and Brier are **exact sums over the target law** with no evaluation Monte Carlo noise -- the role Gauss-Hermite quadrature plays for a normal eta.  The whole study was also re-run under a normal eta law with 80-point Gauss-Hermite quadrature purely as an implementation reference (Table 2g).

### 2a. The staging contributes nothing

The first thing to record is structural, not empirical.  Arm C uses the full penalty grid at every n; arm D uses the full grid whenever n >= 50.  **For n >= 50 the two arms are therefore the same estimator by construction**, and the design's only n below 50 is n = 25, where the staged rule merely deletes rho = 0.01 from the grid (n = 25 is above the rule's own n < 20 cut, so the slope is never fixed by staging anywhere in this design).  With the cross-validation folds shared between C and D within a replicate -- which the brief requires, and which isolates the grid from CV noise -- the consequences are:

| quantity | value |
|---|---|
| mean LL(D) - LL(C), all 225 cells | -0.000355 |
| mean LL(D) - LL(C), n >= 50 (180 cells) | 0.000000 |
| largest absolute LL(D) - LL(C) over cells | 0.012432 |
| cells where the difference exceeds 2 Monte Carlo SE | 10 |
| share of replicates where C and D select the same penalty | 0.997404 |
|   the same, restricted to n >= 50 | 1 |
|   the same, restricted to n = 25 | 0.987022 |

With an **independent** fold draw for D, which is what the original code did and what makes the number comparable with the brief's pilot, the mean difference at n >= 50 is -0.000019 (pilot: -0.00004) and the same penalty is chosen in 91.2% of replicates (pilot: 66%); 8 cells then exceed twice their Monte Carlo SE, which is CV resampling noise, not staging.

So the staging is either exactly inert (n >= 50) or worth a mean 0.000355 in expected log loss (n = 25).  There is nothing for it to contribute.

### 2b. Information spread at fixed n

| n | design cells | min n_eff | max n_eff | fold spread | min design-cell mean | max design-cell mean | fold spread of cell means |
|---|---|---|---|---|---|---|---|
| 25 | 45 | 0.01 | 7.84 | 850 | 0.04 | 5.32 | 137 |
| 50 | 45 | 0.03 | 14.34 | 499 | 0.08 | 10.75 | 140 |
| 100 | 45 | 0.07 | 27.48 | 382 | 0.15 | 21.75 | 142 |
| 200 | 45 | 0.23 | 53.02 | 227 | 0.31 | 43.50 | 140 |
| 400 | 45 | 0.50 | 98.93 | 199 | 0.62 | 87.21 | 140 |

At **every** labelled sample size the available calibration information spans two or more orders of magnitude, and the ranges overlap heavily across n: at n = 25 n_eff reaches 7.84, while at n = 400 it falls as low as 0.50.  Ranked by design-cell means the spread is essentially constant at about 140-fold at every n.  A cut point on n cannot separate the cells that support a slope from those that do not.  See `fig_information_spread.png`.

### 2c. Where adaptation starts to pay

| S range | cells | B beats A | E beats A | mean LL(B) - LL(A) | mean LL(E) - LL(A) | mean n_eff |
|---|---|---|---|---|---|---|
| [0, 0.5) | 117 | 1/117 | 5/117 | +0.0139 | +0.0036 | 5.17 |
| [0.5, 1) | 21 | 3/21 | 4/21 | +0.0086 | +0.0026 | 7.28 |
| [1, 2) | 24 | 7/24 | 9/24 | +0.0125 | +0.0021 | 10.52 |
| [2, 4) | 20 | 10/20 | 12/20 | +0.0050 | -0.0000 | 15.99 |
| [4, 8) | 17 | 12/17 | 13/17 | +0.0059 | -0.0047 | 22.96 |
| [8, 16) | 12 | 11/12 | 12/12 | -0.0086 | -0.0095 | 19.41 |
| [16, 32) | 9 | 9/9 | 9/9 | -0.0156 | -0.0154 | 28.67 |
| [32, 64) | 4 | 4/4 | 4/4 | -0.0269 | -0.0268 | 44.37 |
| [64, inf) | 1 | 1/1 | 1/1 | -0.0471 | -0.0471 | 86.71 |

Ordering cells by S = (b0 - 1)^2 * n_eff produces a clean and monotone crossover.  Unrestricted two-parameter ML (B) is worse than intercept-only updating in the mean until **S is between 4 and 8**, and beats it in almost every cell once S >= 8.  The information-adaptive arm (E) crosses earlier, between **S = 2 and S = 4**, and is never much worse than A below the crossover: its largest mean penalty in any S bin is +0.0036 against +0.0139 for B.  Above S = 8 the two coincide, because E's positive-part penalty goes to zero once the observed slope shift is large relative to its standard error.  See `fig_crossover.png`.

### 2d. n against n_eff as a criterion

| criterion for predicting 'unrestricted two-parameter ML beats intercept-only updating' | AUC |
|---|---|
| AUC of n_eff | 0.9157 |
| AUC of n | 0.7927 |
| AUC of S = (b0-1)^2 n_eff (reference; uses the unobservable b0) | 0.9566 |
| AUC of n_eff, restricted to b0 != 1 | 0.9477 |
| AUC of n, restricted to b0 != 1 | 0.8205 |

Over all 225 cells (58 positives, 167 negatives) n_eff attains AUC **0.9157** against **0.7927** for n; restricted to the 180 cells with a genuine slope shift the gap widens to 0.9477 against 0.8205.  The brief's pilot reported 0.937 against 0.881 -- the same ordering, with a larger margin here because the project DGP's skewed eta law makes n a worse proxy than it is under a normal law.  Both quantities are observable before any calibration parameter is estimated; n_eff is simply the better one.

The same table as 2c, keyed on n instead of S:

| n | cells | B beats A | E beats A | mean LL(B) - LL(A) | mean LL(E) - LL(A) | mean n_eff |
|---|---|---|---|---|---|---|
| n = 25 | 45 | 0/45 | 1/45 | +0.0391 | +0.0091 | 1.73 |
| n = 50 | 45 | 4/45 | 5/45 | +0.0088 | +0.0014 | 3.51 |
| n = 100 | 45 | 11/45 | 12/45 | +0.0010 | -0.0019 | 7.08 |
| n = 200 | 45 | 19/45 | 22/45 | -0.0024 | -0.0035 | 14.20 |
| n = 400 | 45 | 24/45 | 29/45 | -0.0040 | -0.0044 | 28.48 |

Keyed on n the transition is smeared: B still loses on average at n = 100 and only turns positive at n = 200, and even at n = 400 it fails in 21 of 45 cells.  Keyed on S the same 225 cells sort almost perfectly.

### 2e. Risk envelope

| arm | mean excess | max excess | median excess | cells within 0.001 of the oracle |
|---|---|---|---|---|
| A  intercept-only | 0.0024 | 0.0471 | 0.0000 | 177 |
| B  unrestricted ML | 0.0109 | 0.1650 | 0.0031 | 68 |
| C  CV penalty, no staging | 0.0025 | 0.0313 | 0.0003 | 141 |
| D  staged rule (comparator) | 0.0021 | 0.0225 | 0.0002 | 145 |
| D' staged rule, independent CV folds | 0.0021 | 0.0230 | 0.0002 | 150 |
| E  information-adaptive | 0.0026 | 0.0246 | 0.0006 | 136 |

Excess is measured against the per-cell oracle min(LL_A, LL_B).  The unrestricted slope is the clear loser: mean excess 0.0109 and worst case 0.1650, four to five times the mean risk of any other arm.  Arms A, C, D and E all sit within 0.0026 of the oracle in the mean.  The difference between them is in the tail: intercept-only updating has the **worst maximum** excess of the four (0.0471), and the information-adaptive arm roughly halves it (0.0246) at the cost of a slightly larger mean (0.0026 against 0.0024).  The brief's pilot found E ahead of A on the mean as well (0.0025 against 0.0058); under the project DGP's skewed eta law A is penalised less often, so E's advantage shows up in the worst case rather than the average.

### Arm diagnostics

| arm | mean expected LL | mean expected Brier | median MC SE | median SD of fitted slope | max SD of fitted slope | mean share fixing b = 1 |
|---|---|---|---|---|---|---|
| A  intercept-only | 0.4100 | 0.1271 | 0.000362 | 0.0000 | 0.0000 | 1.0000 |
| B  unrestricted ML | 0.4185 | 0.1273 | 0.000692 | 0.7705 | 6.6993 | 0.0000 |
| C  CV penalty, no staging | 0.4101 | 0.1268 | 0.000437 | 0.1763 | 1.7036 | 0.9239 |
| D  staged rule (comparator) | 0.4097 | 0.1268 | 0.000437 | 0.1666 | 1.0180 | 0.9246 |
| D' staged rule, independent CV folds | 0.4097 | 0.1268 | 0.000433 | 0.1810 | 0.9691 | 0.9243 |
| E  information-adaptive | 0.4102 | 0.1266 | 0.000556 | 0.3638 | 2.4173 | 0.5656 |

One-class calibration subsets occur in 2.39% of replicates overall, reaching 32.6% in the worst cell; by n they are 25: 9.24%, 50: 2.54%, 100: 0.19%, 200: 0.00%, 400: 0.00%.  The unrestricted ML fit fails to converge in 7.85% of replicates (all at small n with a near-separated subset, where the slope is driven to the +/-25 clip).  n_eff over all 112,500 replicates spans 0.0092 to 98.93.

### Table 2g -- the same study under a normal eta law (implementation reference)

| quantity | project DGP eta law | normal eta law | brief's pilot reference |
|---|---|---|---|
| mean LL(D) - LL(C), n >= 50, shared folds | 0.000000 | 0.000000 | -0.00004 |
| mean LL(D') - LL(C), n >= 50, independent folds | -0.000019 | -0.000065 | -0.00004 |
| same penalty selected, independent folds | 0.912436 | 0.869796 | 0.66 |
| AUC of n_eff | 0.915651 | 0.882955 | 0.937 |
| AUC of n | 0.792690 | 0.839431 | 0.881 |
| mean oracle excess, arm E | 0.002564 | 0.002996 | 0.0025 |
| mean oracle excess, arm A | 0.002406 | 0.004629 | 0.0058 |
| mean oracle excess, arm B | 0.010904 | 0.025394 | 0.0244 |

Under the normal law n_eff at n = 50 spans 0.09 to 19.19 against the pilot's 0.18 to 13.95 -- the upper end agrees to within 38%.  Every headline quantity moves toward the pilot when the eta law is switched to normal, which is the expected behaviour and locates the remaining differences in the eta law rather than in the implementation.

### Verification: seed stability and Monte Carlo error

| comparison | cells re-run | sign unchanged | share | largest change in the difference |
|---|---|---|---|---|
| LL(B) - LL(A) | 43 | 42 | 0.9767 | 0.031266 |
| LL(E) - LL(A) | 43 | 43 | 1.0000 | 0.009048 |
| LL(C) - LL(A) | 43 | 39 | 0.9070 | 0.018187 |
| LL(D) - LL(C) | 43 | 41 | 0.9535 | 0.006133 |

A stratified subset of **43 design cells** -- every level of n, every level of b0, and both extremes of prevalence and prediction spread -- was re-run with a fresh seed (20260926) at 500 replicates.  **165 of 172 headline comparisons keep their sign (95.9%).**  The median per-cell Monte Carlo SE of expected log loss is 0.000725 in the main run and 0.000691 in the re-run (largest across arms within each cell).

## 3. Link comparison on the battery corpus

The two quantities the brief quotes for the existing cloglog model come from two different existing protocols, so both were rerun per link with nothing else changed:

* **coefficient** -- the 493-cell / 16,538-row pool under the Study C protocol of `79_mcsm_experiment_suite.py`: standardised `log_interval_mid`, `lag_soh`, `delta_soh_10` plus dataset and chemistry dummies, penalty alpha = 0.03162277660168379 held fixed across links, 1,000 whole-cell bootstrap resamples, seed 20260823;
* **prediction** -- the leakage-free rolling-origin protocol that produced the quoted ranges: `event ~ C(batch) + lag_soh + delta_soh_10` on the MATR official-endpoint cycle path, published whole-cell train/test split, nine landmark-horizon settings, complete-case labelling.

The cloglog row reproduces the published values exactly (hazard ratio 0.499526, interval (0.471587, 0.524324); Brier 0.0366 - 0.0967, AUC 0.9174 - 0.9949), which validates both pipelines before the link is changed.

### Table 3 -- link comparison

| link | lagged-SOH coefficient | 95% bootstrap interval | exp(coef) | 95% interval for exp(coef) | mean Brier | mean AUC | Brier range | AUC range |
|---|---|---|---|---|---|---|---|---|
| cloglog | -0.6941 | (-0.7517, -0.6456) | 0.4995 | (0.4716, 0.5243) | 0.0732 | 0.9616 | 0.0366 - 0.0967 | 0.9174 - 0.9949 |
| logit | -0.7038 | (-0.7611, -0.6533) | 0.4947 | (0.4672, 0.5203) | 0.0722 | 0.9678 | 0.0319 - 0.0967 | 0.9174 - 0.9949 |
| probit | -0.7100 | (-0.7704, -0.6820) | 0.4917 | (0.4628, 0.5056) | 0.0722 | 0.9670 | 0.0323 - 0.0969 | 0.9174 - 0.9949 |

**This task does not reproduce on real data what the brief expected, and the honest result is the opposite of the anticipated one.**  The ratio of the largest to the smallest exp(coef) across links is **1.0160** -- a 1.6% disagreement -- beside a spread in mean Brier of 0.0010 (0.0722 to 0.0732) and in mean AUC of 0.0062 (0.9616 to 0.9678).  The three links agree on the coefficient just as closely as they agree on prediction.  The bootstrap intervals overlap almost completely.

**Why.**  The comparison is only well posed at the manuscript's selected penalty.  Refitting all three links across the penalty grid:

| penalty alpha | cloglog exp(coef) | logit exp(coef) | probit exp(coef) | max/min | all three links converged? |
|---|---|---|---|---|---|
| 0 | 0.37690 | 0.00005 | 0.00157 | 7915.88 | no (cloglog) |
| 0.0001 | 0.37690 | 0.00121 | 0.00964 | 310.86 | no (cloglog) |
| 0.001 | 0.37690 | 0.03170 | 0.05661 | 11.89 | no (cloglog) |
| 0.01 | 0.37690 | 0.25740 | 0.29695 | 1.46 | no (cloglog) |
| 0.0316228 | 0.49953 | 0.49470 | 0.49165 | 1.02 | yes |
| 0.1 | 0.74183 | 0.74177 | 0.66669 | 1.11 | yes |

At every penalty weaker than the selected one **the cloglog fit does not converge** (L-BFGS-B stops after one iteration with a non-zero gradient), while logit and probit converge to coefficients that march off toward a separation boundary -- logit exp(coef) = 0.00005 at alpha = 0.  The apparent thousand-fold link disagreement in that regime is a comparison between a failed fit and a diverging one, and supports nothing.

The cause is threshold proximity, which the manuscript already discusses in another context: the endpoint is defined by SOH crossing 0.80 and the covariate is lagged SOH measured immediately before it.  Lagged SOH nearly separates the interval event -- univariate AUC **0.9850**, all 303 event rows confined to lag_soh in [0.7953, 0.8671], an event rate of 32.6% below lag_soh = 0.8214 against 0.298% above it, against an overall row event rate of 1.83%.  Under near-separation the unpenalised estimates are not defined, which is exactly why five-fold whole-cell CV selected so strong a penalty; and a penalty strong enough to stabilise the fit also dominates the likelihood enough to pull all three links to a common shrunken value.

**What the paper can and cannot claim from this.**  It can claim the prediction half: at a fixed grouping, changing the binary link moves leakage-free rolling-origin Brier by 0.0010 and AUC by 0.0062 across nine landmark-horizon settings, i.e. not at all in any operational sense.  It **cannot** use this corpus to show that only the cloglog coefficient stays on the generating proportional-hazards scale: at the one penalty where all three links are estimable they agree to 1.6%, so there is no real-data coefficient disagreement to point at.  Recommend reporting the prediction invariance as the real-data fact and leaving the coefficient-scale argument to the simulation, where the generating coefficient is known.

## 4. Stochastic-process degradation baselines

Degradation is D(t) = 1 - SOH(t) and the persistent 80% SOH endpoint is a threshold at D* = 0.20; this was verified against the data rather than assumed (`eol_cycle` equals the first cycle with SOH <= 0.80 for all 83 MATR events).  Paths live on the project's own 25-cycle observation schedule.  The Wiener process is fitted on the raw series; the gamma process requires a non-decreasing path, so a running maximum is applied to D for it -- **17.4% of raw increments on the MATR cycle path are negative** (median 16.7% per cell, maximum 47.5%), so that step removes a great deal of real capacity regeneration.  Both models are fitted on training cells only and predict each test cell using only its observations at or before L.

All four fairness requirements hold.  Requirement 4 is the one the published protocol does not enforce: one MATR test cell reaches the endpoint at cycle 148, before every landmark, and `make_landmark_table` retains it.  The headline table excludes it for **every** model, and every model is scored on exactly the cells all models can score, so no model gets a different denominator.

### Table 4 -- MATR official-endpoint path, raw cycle scale, whole-cell split, requirement 4 enforced, full training window

| model | L | H | n evaluated | events | Brier | log loss | AUC |
|---|---|---|---|---|---|---|---|
| wiener | 150 | 500 | 41 | 13 | 0.2807 | 0.9805 | 0.6126 |
| wiener | 150 | 800 | 38 | 19 | 0.2385 | 0.6716 | 0.6427 |
| wiener | 150 | 1000 | 32 | 22 | 0.2392 | 0.7018 | 0.4545 |
| wiener | 200 | 500 | 41 | 13 | 0.2905 | 1.3170 | 0.7280 |
| wiener | 200 | 800 | 38 | 19 | 0.2405 | 0.6651 | 0.7424 |
| wiener | 200 | 1000 | 32 | 22 | 0.2371 | 0.6761 | 0.5591 |
| wiener | 300 | 500 | 41 | 13 | 0.2946 | 2.3250 | 0.8379 |
| wiener | 300 | 800 | 38 | 19 | 0.2601 | 0.7294 | 0.8615 |
| wiener | 300 | 1000 | 32 | 22 | 0.2336 | 0.6628 | 0.7000 |
| gamma | 150 | 500 | 41 | 13 | 0.2872 | 1.0791 | 0.6126 |
| gamma | 150 | 800 | 38 | 19 | 0.2967 | 0.8091 | 0.6427 |
| gamma | 150 | 1000 | 32 | 22 | 0.3160 | 0.9275 | 0.4545 |
| gamma | 200 | 500 | 41 | 13 | 0.2924 | 1.4245 | 0.7280 |
| gamma | 200 | 800 | 38 | 19 | 0.3351 | 0.9737 | 0.7424 |
| gamma | 200 | 1000 | 32 | 22 | 0.3535 | 1.0668 | 0.5591 |
| gamma | 300 | 500 | 41 | 13 | 0.2706 | 1.9499 | 0.8489 |
| gamma | 300 | 800 | 38 | 19 | 0.3474 | 1.2983 | 0.8643 |
| gamma | 300 | 1000 | 32 | 22 | 0.3642 | 1.4271 | 0.7045 |
| cloglog hazard model | 150 | 500 | 41 | 13 | 0.0673 | 0.2177 | 0.9698 |
| cloglog hazard model | 150 | 800 | 38 | 19 | 0.0758 | 0.2213 | 0.9834 |
| cloglog hazard model | 150 | 1000 | 32 | 22 | 0.0982 | 0.3452 | 0.9182 |
| cloglog hazard model | 200 | 500 | 41 | 13 | 0.0583 | 0.1955 | 0.9890 |
| cloglog hazard model | 200 | 800 | 38 | 19 | 0.0863 | 0.2484 | 0.9695 |
| cloglog hazard model | 200 | 1000 | 32 | 22 | 0.0997 | 0.6364 | 0.9136 |
| cloglog hazard model | 300 | 500 | 41 | 13 | 0.0468 | 0.5845 | 0.9602 |
| cloglog hazard model | 300 | 800 | 38 | 19 | 0.0691 | 0.2862 | 0.9806 |
| cloglog hazard model | 300 | 1000 | 32 | 22 | 0.0843 | 0.7908 | 0.9205 |
| random forest | 150 | 500 | 41 | 13 | 0.0497 | 0.1902 | 0.9698 |
| random forest | 150 | 800 | 38 | 19 | 0.0705 | 0.2355 | 0.9806 |
| random forest | 150 | 1000 | 32 | 22 | 0.0767 | 0.2507 | 0.9545 |
| random forest | 200 | 500 | 41 | 13 | 0.0422 | 0.1534 | 0.9780 |
| random forest | 200 | 800 | 38 | 19 | 0.0602 | 0.2062 | 0.9861 |
| random forest | 200 | 1000 | 32 | 22 | 0.0718 | 0.2277 | 0.9591 |
| random forest | 300 | 500 | 41 | 13 | 0.0393 | 0.1276 | 0.9890 |
| random forest | 300 | 800 | 38 | 19 | 0.0572 | 0.1953 | 0.9834 |
| random forest | 300 | 1000 | 32 | 22 | 0.0744 | 0.2322 | 0.9591 |

Means across the nine settings:

| model | mean Brier | mean log loss | mean AUC | mean predicted risk | mean observed rate |
|---|---|---|---|---|---|
| wiener | 0.2572 | 0.9699 | 0.6821 | 0.3271 | 0.5015 |
| gamma | 0.3181 | 1.2173 | 0.6841 | 0.2189 | 0.5015 |
| cloglog hazard model | 0.0762 | 0.3918 | 0.9561 | 0.5064 | 0.5015 |
| random forest | 0.0602 | 0.2021 | 0.9733 | 0.5010 | 0.5015 |

The first-passage models are **three to four times worse on Brier** than the hazard model and lose about 0.27 of AUC.  The reason is visible in the last two columns: they systematically under-predict, placing a mean risk of 0.3271 (Wiener) and 0.2189 (gamma) against an observed rate of 0.5015, while both the hazard model and the random forest are calibrated in the mean to within 0.0049.  The random forest is the best of the four on this corpus, which the paper already reports for the hazard-versus-ML comparison and which is not affected here.

### Fitted parameters and whether they are physically sensible

| corpus | train window | mu0 (per cycle) | s2mu | sigma^2 | implied mean cycles to D* = 0.20 |
|---|---|---|---|---|---|
| matr | full | 2.9713e-04 | 1.8829e-08 | 3.0673e-06 | 674 |
| matr | truncH | 2.5691e-04 | 2.5693e-08 | 3.1496e-06 | 795 |
| sixsource | full | 1.2152e-01 | 4.3171e-17 | 1.6866e+01 | 885 |
| sixsource | truncH | 2.1671e-04 | 3.9088e-17 | 5.9469e-05 | 925 |

| corpus | train window | alpha | a0 | b0 | mean rate (per cycle) | implied mean cycles to D* = 0.20 |
|---|---|---|---|---|---|---|
| matr | full | 0.0402 | 2.7347 | 1.5412e-02 | 2.2640e-04 | 883 |
| matr | truncH | 0.0616 | 1.2527 | 2.7313e-03 | 1.1678e-04 | 1911 |
| sixsource | full | 0.0839 | 1.9132 | 4.5720e-03 | 2.0045e-04 | 998 |
| sixsource | truncH | 0.1149 | 1.2642 | 1.7169e-03 | 1.4121e-04 | 1449 |

On MATR the parameters are physically sensible in scale -- a mean drift of 2.97e-04 of SOH per cycle, reaching D* = 0.20 at about cycle 674 for the Wiener model and cycle 883 for the gamma process -- but **both under-state the degradation rate**, and that is the whole result.  The observed median endpoint among the 83 MATR events is cycle **527** (10th to 90th percentile 457 to 910), so the Wiener model is late by about 147 cycles and the gamma process by about 356.

The reason is that lithium-ion capacity fade is strongly **convex in cycle number** while both models assume time-homogeneous increments.  Measured directly on the 81 event cells with usable paths, the degradation accrued by a given fraction of observed life, relative to the proportional amount, is 0.448 at a quarter of life, 0.359 at half and 0.459 at three quarters (medians).  A constant-drift Wiener process and a constant-rate gamma process both fit that early flat region and then extrapolate linearly, so at a landmark of 150 to 300 cycles -- where median D is only about 0.03 against a threshold of 0.20 -- they place almost no probability on crossing by cycle 500 to 1000.

**Time scale.**  The headline uses raw cycle number.  Dividing cycle number by the per-source median observed life is, for a single-source corpus, a pure change of units: the fitted parameters rescale and the predictions are invariant.  That was confirmed numerically rather than assumed -- the largest Brier difference between the raw and normalised runs across all model x setting combinations is 8.436e-09.  A normalised time scale can only matter where the scale differs across sources, which is the six-source corpus below.

**Power-transformed time, as a diagnostic.**  To test whether the failure is the first-passage framework or specifically the assumption of time-homogeneous increments, both models were refitted on a power time scale t -> t^gamma with gamma profiled over the Wiener likelihood.  This is an extension beyond the two models the brief specifies and is reported only as a diagnostic.  It helps substantially and is still not close: mean Brier improves from 0.2572 to 0.2391 for the Wiener model and from 0.3181 to 0.2439 for the gamma process, against 0.0762 for the hazard model.  So convexity in cycle number is the dominant part of the gap, but not all of it.

### Table 4b -- six-source corpus (493 cells), five-fold whole-cell cross-validation, raw cycle scale

This extends the comparison to the full heterogeneous corpus.  Read it with two warnings, both established below rather than assumed.

* The **Wiener rows here come from an unidentified fit** and must not be read as a model ranking.  On this representation the likelihood has a spurious high-likelihood region: the fold-averaged estimates include mu0 = 0.1215 of SOH per cycle and sigma^2 = 16.87, which are physically impossible, while the drift-variance component collapses to about 4e-17.  The gamma and hazard rows are from converged fits with physically sensible parameters.
* The row labelled *cloglog cumulative-risk (artefact-affected)* is the accumulation 1 - prod(1 - h_ij) used by the published transport analysis.  It is shown only to quantify the survivorship artefact described in section 5, and is not the hazard-model comparator.  The comparator row *cloglog hazard model* is a landmark-style fit on the fixed-horizon label with no accumulation, which is what the MATR arm uses.

| model | L | H | n evaluated | events | Brier | log loss | AUC |
|---|---|---|---|---|---|---|---|
| wiener | 150 | 500 | 422 | 116 | 0.2756 | 1.8252 | 0.6647 |
| gamma | 150 | 500 | 422 | 116 | 0.2087 | 1.5466 | 0.7913 |
| cloglog hazard model | 150 | 500 | 422 | 116 | 0.1381 | 0.4142 | 0.8275 |
| cloglog cumulative-risk (artefact-affected) | 150 | 500 | 422 | 116 | 0.2039 | 0.6230 | 0.6351 |
| wiener | 150 | 800 | 381 | 178 | 0.2739 | 1.6681 | 0.7048 |
| gamma | 150 | 800 | 381 | 178 | 0.1650 | 1.7567 | 0.8545 |
| cloglog hazard model | 150 | 800 | 381 | 178 | 0.1235 | 0.4144 | 0.9029 |
| cloglog cumulative-risk (artefact-affected) | 150 | 800 | 381 | 178 | 0.2814 | 0.8089 | 0.5546 |
| wiener | 150 | 1000 | 346 | 198 | 0.2588 | 1.4660 | 0.7268 |
| gamma | 150 | 1000 | 346 | 198 | 0.1490 | 1.6344 | 0.8720 |
| cloglog hazard model | 150 | 1000 | 346 | 198 | 0.0929 | 0.3864 | 0.9372 |
| cloglog cumulative-risk (artefact-affected) | 150 | 1000 | 346 | 198 | 0.3222 | 0.9106 | 0.4879 |
| wiener | 200 | 500 | 405 | 98 | 0.2651 | 1.6989 | 0.6939 |
| gamma | 200 | 500 | 405 | 98 | 0.1859 | 1.5580 | 0.7888 |
| cloglog hazard model | 200 | 500 | 405 | 98 | 0.1299 | 0.3857 | 0.8419 |
| cloglog cumulative-risk (artefact-affected) | 200 | 500 | 405 | 98 | 0.1717 | 0.5173 | 0.7379 |
| wiener | 200 | 800 | 365 | 156 | 0.2594 | 1.6112 | 0.7417 |
| gamma | 200 | 800 | 365 | 156 | 0.1679 | 1.8173 | 0.8668 |
| cloglog hazard model | 200 | 800 | 365 | 156 | 0.1128 | 0.3937 | 0.9270 |
| cloglog cumulative-risk (artefact-affected) | 200 | 800 | 365 | 156 | 0.2610 | 0.7458 | 0.5931 |
| wiener | 200 | 1000 | 330 | 176 | 0.2485 | 1.4515 | 0.7646 |
| gamma | 200 | 1000 | 330 | 176 | 0.1581 | 1.6742 | 0.8864 |
| cloglog hazard model | 200 | 1000 | 330 | 176 | 0.0782 | 0.3454 | 0.9573 |
| cloglog cumulative-risk (artefact-affected) | 200 | 1000 | 330 | 176 | 0.3036 | 0.8518 | 0.5042 |
| wiener | 300 | 500 | 398 | 87 | 0.2523 | 1.6067 | 0.7395 |
| gamma | 300 | 500 | 398 | 87 | 0.1356 | 1.4354 | 0.8247 |
| cloglog hazard model | 300 | 500 | 398 | 87 | 0.1022 | 0.3479 | 0.8894 |
| cloglog cumulative-risk (artefact-affected) | 300 | 500 | 398 | 87 | 0.1665 | 0.5086 | 0.7310 |
| wiener | 300 | 800 | 358 | 145 | 0.2403 | 1.5611 | 0.7834 |
| gamma | 300 | 800 | 358 | 145 | 0.1524 | 1.5885 | 0.9195 |
| cloglog hazard model | 300 | 800 | 358 | 145 | 0.0725 | 0.3405 | 0.9573 |
| cloglog cumulative-risk (artefact-affected) | 300 | 800 | 358 | 145 | 0.2693 | 0.7798 | 0.5254 |
| wiener | 300 | 1000 | 323 | 165 | 0.2367 | 1.4335 | 0.8053 |
| gamma | 300 | 1000 | 323 | 165 | 0.1500 | 1.3639 | 0.9458 |
| cloglog hazard model | 300 | 1000 | 323 | 165 | 0.0446 | 0.3058 | 0.9774 |
| cloglog cumulative-risk (artefact-affected) | 300 | 1000 | 323 | 165 | 0.3170 | 0.9023 | 0.4432 |

Means across the nine settings: hazard model Brier 0.0994 / AUC 0.9131, gamma 0.1636 / 0.8611, Wiener (unidentified) 0.2567 / 0.7361, and the artefact-affected accumulation 0.2552 / 0.5791.  The ordering agrees with the MATR arm wherever it is trustworthy: the hazard model is ahead on both Brier and AUC.  The gamma process does markedly better here than on MATR (Brier 0.1636 against 0.3181) but its log loss is 1.5972 against 0.3704 for the hazard model -- it is confidently wrong on the cells it gets wrong.

**Why the six-source representation is hostile to a first-passage model.**  The pool's grouped representation gives D = 1 - `lag_soh`, and `lag_soh` there runs from 0.2338 to 1.6825, so D ranges from **-0.6825 to +0.7662**: 311 of 16,538 rows (1.9%) have negative degradation, spread over **109 of the 493 cells**, and 28 rows already sit beyond the D* = 0.20 threshold.  Both processes require a non-negative degradation path, and the Wiener model additionally assumes D(0) = 0 at t = 0 while every pool path begins at interval start 100.  The MATR cycle path by contrast has D between 0.0002 and 0.2524 with **no** negative rows, which is why the MATR arm identifies cleanly and carries the headline.  A reviewer asking for the six-source comparison should be told that the corpus first needs a degradation representation built for it.

### Sensitivity

Mean Brier by training window, requirement-4 variant and time scale (MATR).  Enforcing requirement 4 changes the hazard model's mean Brier from 0.0752 to 0.0762, i.e. the one prematurely-crossed cell matters little here but the comparison is now exact.  Truncating the training paths at the horizon makes both process models slightly worse (Wiener 0.2572 -> 0.2656, gamma 0.3181 -> 0.3471), as expected, since it removes the late accelerating part of the training paths.

```
window                  full                                          truncH                                
variant                 fair                    published               fair            published           
timescale              cycle normalised   power     cycle normalised   cycle normalised     cycle normalised
model                                                                                                       
cloglog hazard model  0.0762     0.0762  0.0762    0.0752     0.0752  0.0762     0.0762    0.0752     0.0752
gamma                 0.3181     0.3181  0.2439    0.3180     0.3180  0.3471     0.3471    0.3470     0.3470
random forest         0.0602     0.0602  0.0602    0.0604     0.0604  0.0602     0.0602    0.0604     0.0604
wiener                0.2572     0.2572  0.2391    0.2576     0.2576  0.2656     0.3087    0.2655     0.2655
```

**What agrees and what diverges.**  The expected story in the brief was that a first-passage model would be competitive when degradation is close to monotone and the threshold is far away, and would degrade where regeneration or protocol heterogeneity breaks the process assumptions.  What the data actually show is sharper and different: the first-passage models are **worst exactly where the threshold is far away**.  At L = 300, H = 500 -- the setting with the most margin, where median D is about 0.03 against a threshold of 0.20 -- the Wiener model predicts a mean risk of 0.0222 against an observed rate of 0.3171 and records a log loss of 2.3250, its worst of the nine.  They are relatively least bad at the long horizons, where enough degradation has accrued that even a linear extrapolation reaches the threshold.  The mechanism is the convexity of capacity fade, not regeneration: the running maximum needed by the gamma process removes 17.4% of increments and the gamma process is still the weaker of the two on MATR.  Discrimination survives better than calibration -- Wiener AUC reaches 0.8615 at L = 300, H = 800 -- so the processes do rank cells, they just cannot put the risk on the right scale.  The hazard model absorbs both the convexity, through interval-specific baselines, and the regeneration, through lagged SOH and recent SOH change as covariates.

## 5. Anything that failed, was skipped, or looks wrong

**1. Task 3 does not deliver the finding the brief anticipated.**  The brief expected the links to "disagree materially on the coefficient while agreeing on prediction".  On this corpus they agree on both, to 1.6% on exp(coef).  The mechanism is documented in section 3 with a penalty sweep and separation diagnostics: at any penalty weak enough to let the links diverge, the cloglog fit does not converge and the logit and probit fits run off toward a separation boundary, because lagged SOH nearly separates the interval event (univariate AUC 0.9850).  Reported as found rather than presented as a confirmation.

**2. A corpus-consistent version of Table 3 was attempted and rejected as invalid.**  Task 3's method section asks for the 493-cell corpus, but the quoted Brier and AUC ranges come from the MATR path, so a 493-cell five-fold whole-cell version was also built.  It must not be used.  Cell level fixed-horizon risk on the pool is accumulated as 1 - prod(1 - h_ij) over intervals in [L, H], and a cell that fails leaves the risk set, so it contributes fewer intervals: at L = 150, H = 800 failing cells contribute 12.4 intervals on average against 27.0 for survivors, and **the AUC of the interval count alone is 0.0000** -- perfectly inversely predictive.  The accumulated risk mixes the hazard signal with that artefact, and the mixture is link-dependent: the correlation between accumulated risk and interval count is +0.146 for cloglog and -0.366 for probit, which drags cloglog AUC down to 0.557 and inflates probit's to 0.920.  The apparent probit advantage is survivorship arithmetic, not a link effect.  Numbers are in `results/t3_link_prediction.csv` under `arm = pool_wholecell_cv`, flagged there, and are not used anywhere in this report.  **The same accumulation is used by the published transport analysis behind Tables 8 and 9**; its before-versus-after log-loss comparisons are unaffected because both sides share the artefact, but any discrimination claim from those tables should be re-checked.

**3. The brief's Task 3 specification is internally inconsistent.**  It asks for the 493-cell person-period corpus "exactly as used for the main results" and then quotes a hazard ratio that comes from the 493-cell pool and Brier/AUC ranges that come from the 139-cell MATR official-endpoint path.  Both protocols were run per link and both reproduce their published reference values; Table 3 pairs them as the brief's own table does, and the provenance of each column is stated.

**4. Two defects in the reproducibility archive.**  `src/pipeline/79_mcsm_experiment_suite.py` sets `ROOT = Path(__file__).resolve().parents[1]`, which resolves to `<root>/src` after the archive move, so `load_pool()` raises FileNotFoundError from a clean checkout; and `src/pipeline/57_phase2_expanded_transport.py`, which `74_transport_recalibration_loco.py` imports at module load, is absent from the archive entirely.  Both were repaired locally (the first by routing through `project_paths`, the second by copying the file from the working tree and patching its hard-coded `D:/discrel` root).  **These should be fixed in the archive before it is deposited**, because stage 03 and stage 08 cannot run without them.

**5. Descriptive discrepancy in the recalibration study's size.**  The brief and the supplementary text describe "900 design cells x 500 replicates".  The grid actually coded in `83_study_b_shrunk_recalibration.py` is 2 target sizes x 4 baseline shifts x 3 slope scalings x 7 or 8 calibration schemes = 180 design cells, evaluated for 5 methods, i.e. 900 design-cell-by-method rows.  Nothing turns on this, but the manuscript wording overstates the number of design cells by a factor of five.

**6. Wiener convergence flags are unreliable and were checked another way.**  `fit_wiener` frequently reports `converged = False` because Nelder-Mead exhausts its iteration budget after L-BFGS-B has already found the optimum.  Every headline fit was therefore verified by perturbation: moving each parameter by +/-2% and +/-20% never lowered the objective.  The flag is reported in `results/t4_first_passage_params_*.csv` but should not be read as a failure.

**6b. Two numerical defects were found and fixed in the first-passage fitting, and one of them had reversed a conclusion.**  (a) The gamma fit was not scale-equivariant: the same data expressed in cycles and in normalised cycles gave mean Brier 0.3181 and 0.4298, although the gamma process is closed under a rescaling of time and the predictions must be identical.  Fitting is now done with time standardised internally to a median increment of 1 and alpha transformed back, after which the two agree to 4.7e-09 on real data and to 6.7e-08 across a 1,860-fold change of units on synthetic data.  (b) `fit_wiener` from a single moment start collapsed the drift-variance component onto its lower bound on some six-source folds; profiling the likelihood showed a clear interior optimum with the boundary **429 nll units worse**, so those were optimiser failures, not results.  With multiple starts the MATR numbers are unchanged, but the six-source Wiener Brier moved from 0.1723 to 0.2567 and its AUC from 0.8486 to 0.7361 -- **the earlier, better-looking six-source Wiener result was an artefact of a degenerate fit**.  Had the parameters not been checked for physical plausibility, as the brief instructs, this report would have claimed that a first-passage model beats the hazard model on the six-source corpus.  It does not.

**6c. The six-source first-passage comparison is reported but is not a valid model ranking.**  Even after (6b) the Wiener fit on the pool is unidentified -- fold-averaged mu0 = 0.1215 of SOH per cycle and sigma^2 = 16.87 are physically impossible.  The cause is the representation, not the optimiser: the pool's D = 1 - `lag_soh` ranges from -0.6825 to +0.7662, with 311 of 16,538 rows (1.9%) negative across 109 of the 493 cells and 28 rows already past the D* = 0.20 threshold, while both processes require a non-negative path and the Wiener model assumes D(0) = 0 although every pool path starts at interval 100.  The MATR cycle path has D in [0.0002, 0.2524] with no negative rows, which is why it carries the headline.  The gamma and hazard rows of Table 4b are from converged, physically sensible fits and can be read; the Wiener rows cannot.

**7. Deviations from the reference implementations, all verified.**  The Appendix B code was not executable against real data as supplied and three changes were needed; each is checked in `firstpassage.py::self_test`, which passes.  (a) The Wiener marginal likelihood was rewritten from an O(n^3) Cholesky per path to an O(1) evaluation from parameter-free sufficient statistics, by integrating the drift out analytically; the two agree to 4.3e-14 relative once the reference's omitted 0.5*N*log(2*pi) constant is restored, and the speed-up is what made the study feasible.  (b) The gamma likelihood was vectorised across paths; it agrees with the reference loop to 3.3e-16 relative.  (c) `fit_gamma` is started from method-of-moments values because the reference start (alpha, a0, b0) = (1, 2, 50) implies a degradation rate four orders of magnitude too large, from which the optimiser does not recover.  Two of the reference self-checks were also degenerate as first written -- both the inverse-Gaussian and compound-gamma probabilities evaluated to exactly 1.0000, so they passed while testing nothing; they now sit at 0.4483 and 0.5271 and are compared against Monte Carlo.

**8. Arm C and arm D share cross-validation folds in the headline numbers.**  The brief requires folds to be shared across the penalty grid within a replicate; the supplied reference `select_rho_cv` re-permutes for each candidate, which inflates apparent disagreement between C and D.  Both are reported: shared folds isolate the staging, independent folds reproduce the pilot's 66%-style number.

**9. Environment.**  This machine has 7.35 GB of RAM.  The Task 2 study was killed by the operating system three times before the workload was restructured: BLAS threading is pinned to one thread per process (OpenBLAS allocates per-thread workspace in every worker), the innermost 2x2 solve was replaced by a closed form so LAPACK is not called in the hot loop, and parallelism is by independent sharded single-process jobs rather than a multiprocessing pool, whose result pipe does not survive detachment on this host.  Results are unaffected -- `verify_calib.py` reproduces the same agreement with statsmodels before and after the change -- but the sharding is why results arrive as `t2_recal_project_s{0,1}.json` and are merged.

**10. Not attempted.**  No Firth or bootstrap small-sample correction to the calibration slope was added, and the staged rule appears only as arm D, both as instructed.  Nothing was tuned on test data; all resampling and cross-validation is on whole units.

## 6. Files produced

All paths relative to `F:\discrel\qrei`.

**Scripts**

| path | what it does |
|---|---|
| scripts/calib.py | penalised cloglog recalibration, n_eff, CV-with-1SE selection, empirical-Bayes penalty, staged rule |
| scripts/verify_calib.py | self-test: rho = 0 fit against a statsmodels cloglog GLM, penalised fit against an independent optimiser, n_eff against the GLM slope SE |
| scripts/t1_neff_transport.py | Task 1 -- replays the published transport splits and records n_eff |
| scripts/t2_eta_reservoir.py | Task 2 stage 0 -- derives the target eta law from the project DGP |
| scripts/t2_recal_boundary.py | Task 2 -- the 225-cell, five-arm recalibration study |
| scripts/t2_analyse.py | Task 2 -- items 2a to 2e and two figures |
| scripts/t2_verify.py | Task 2 -- seed-stability re-run and summary |
| scripts/t3_link_comparison.py | Task 3 -- cloglog / logit / probit, coefficient and prediction arms |
| scripts/firstpassage.py | Wiener and gamma first-passage models with self-test |
| scripts/t4_first_passage.py | Task 4 -- the like-for-like comparison |
| scripts/t4_analyse.py | Task 4 -- Table 4 and the figure |
| scripts/build_results.py | generates this report from the result files |

**Results**

* `results/t1_neff_transport_splits.csv`
* `results/t2_cells_normal.csv`
* `results/t2_cells_project.csv`
* `results/t2_eta_nodes.npy`
* `results/t2_eta_nodes_meta.json`
* `results/t2_recal_normal.json`
* `results/t2_recal_normal_s0.json`
* `results/t2_recal_normal_s0.partial.jsonl`
* `results/t2_recal_normal_s1.json`
* `results/t2_recal_normal_s1.partial.jsonl`
* `results/t2_recal_project.json`
* `results/t2_recal_project_s0.json`
* `results/t2_recal_project_s0.partial.jsonl`
* `results/t2_recal_project_s1.json`
* `results/t2_recal_project_s1.partial.jsonl`
* `results/t2_summary_normal.json`
* `results/t2_summary_project.json`
* `results/t2_verify_project_s0.json`
* `results/t2_verify_project_s1.json`
* `results/t2_verify_summary_project.json`
* `results/t3_link_coefficients.csv`
* `results/t3_link_penalty_sensitivity.csv`
* `results/t3_link_prediction.csv`
* `results/t3_meta.json`
* `results/t3_separation_diagnostics.json`
* `results/t4_first_passage_metrics_matr.csv`
* `results/t4_first_passage_metrics_pool.csv`
* `results/t4_first_passage_params_matr.csv`
* `results/t4_first_passage_params_pool.csv`
* `results/t4_meta.json`
* `results/t4_summary.json`

**Figures** (PNG, 320 dpi)

* `figs/fig_crossover.png`
* `figs/fig_information_spread.png`
* `figs/fig_firstpassage.png`

**Seeds**

| what | seed | note |
|---|---|---|
| Task 1 transport split replay | 20260702 + crc32(key) % 100000 | the published base seed; splits reproduce the archive exactly |
| Task 2 eta reservoir | 20260823 | the project's main simulation seed |
| Task 2 recalibration study | SeedSequence([20260825, cell_index]) | the project's recalibration follow-up seed |
| Task 2 seed-stability re-run | SeedSequence([20260926, cell_index]) | fresh, chosen not to collide with any project seed |
| Task 3 bootstrap | SeedSequence([20260823, 0]) | reproduces the published interval exactly |
| Task 4 | 20260907; random forest keeps 20260531 + L + H | the comparator's published per-setting seed |

