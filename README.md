# Validity boundaries of discrete-time hazard models for lithium-ion battery reliability

Reproducibility code and results for the four experiments reported in the
manuscript *Validity Boundaries of Discrete-Time Hazard Models for Lithium-Ion
Battery Reliability* (submitted to *Quality and Reliability Engineering
International*).

`RESULTS.md` is the full generated report: every table, every headline number
and the complete record of what failed or was rejected. It is produced by
`scripts/build_results.py` from the files in `results/`, so the prose and the
numbers cannot drift apart. **Edit the generator, never the markdown.**

## What is here

| Path | Contents |
|---|---|
| `scripts/` | all analysis code (see the table below) |
| `results/` | every table and summary the report is built from |
| `figs/` | figures as generated, named as in `RESULTS.md` |
| `figures/` | the same figures under the manuscript's numbering |
| `RESULTS.md` | the generated report |
| `requirements.txt` | exact package versions used |

Figure numbering, main text:

| Figure | File in `figs/` | Shows |
|---|---|---|
| Figure 1 | *(schematic, not generated here)* | grouped event-time representation and landmark-conditional fixed-horizon prediction |
| Figure 2 | `fig_link_grouping.png` | coefficient bias and empirical 95% coverage by link, against interval width |
| Figure 3 | `fig_information_spread.png` | `n_eff` spans two orders of magnitude at every fixed labelled size |
| Figure 4 | `fig_firstpassage.png` | Brier by landmark-horizon setting, first-passage against hazard model |

`figures/fig1.png` ... `fig4.png` are the same images under those numbers.

Two further figures are generated but not used in the main text:

| File | Shows |
|---|---|
| `figs/fig_crossover.png` | the S = 2-4 adaptation boundary; candidate supplementary figure |
| `figs/fig_transport_neff.png` | `n_eff` over the transport combinations, by held-out unit |

## The four experiments

| Script | Task | What it does |
|---|---|---|
| `calib.py` | shared | penalised complementary log-log recalibration, `n_eff`, CV-with-one-standard-error penalty selection, empirical-Bayes penalty, and the retired staged rule kept only as a comparator |
| `verify_calib.py` | shared | self-test: unpenalised fit against a `statsmodels` cloglog GLM, penalised fit against an independent optimiser, `n_eff` against the GLM slope standard error |
| `t1_neff_transport.py` | 1 | replays the published transport splits and records the calibration information actually available in each |
| `t1_figure.py` | 1 | `fig_transport_neff.png` |
| `t_link_figure.py` | link study | Figure 2, from the archived Study-A link table |
| `t2_eta_reservoir.py` | 2 | derives the target `eta` law from the project's own data-generating process |
| `t2_recal_boundary.py` | 2 | the 225-cell, five-arm recalibration-boundary study |
| `t2_analyse.py` | 2 | items 2a-2e, Figure 3 and `fig_crossover.png` |
| `t2_verify.py` | 2 | seed-stability re-run on a stratified subset |
| `t3_link_comparison.py` | 3 | cloglog / logit / probit, coefficient and prediction arms |
| `firstpassage.py` | 4 | Wiener and gamma first-passage models, with a self-test |
| `t4_first_passage.py` | 4 | the like-for-like comparison against the hazard model |
| `t4_analyse.py` | 4 | Table 4 and Figure 4 |
| `build_results.py` | all | assembles `RESULTS.md` |

## Data

**The battery corpus is not redistributed here.** The analyses read the
harmonised person-period pool
`data/processed/phase2_expanded_person_period_grouped_25.csv`
(16,538 rows, 493 cells, 303 events, six sources: MATR 139, BatteryLife SNL 61,
CALB 27, HUST 77, SDU 70, Tongji 119) and the MATR official-endpoint cycle path
`data/processed/matr_official_person_period_cycle.csv`. Both are derived from
publicly available datasets under their own licences; the harmonisation
pipeline is part of the separate project archive.

Scripts resolve paths through the project archive's
`src/utilities/project_paths.py` and honour two environment variables:

```
CSDA_PROJECT_ROOT   the project archive checkout
CSDA_DATA_ROOT      where data/processed lives (defaults to <root>/data)
```

Tasks 1, 3 and 4 need the data. **Task 2 does not** — it is self-contained
simulation and runs from a clean checkout.

## Reproducing

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Pin BLAS to one thread per process. The studies do only 2x2 linear algebra, so
threaded BLAS buys nothing and its per-thread workspace is what exhausts memory
when several workers run:

```
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
```

Check the two self-tests first. Both must pass before any number is trusted:

```
python scripts/verify_calib.py
python scripts/firstpassage.py
```

Task 2, the only fully self-contained experiment:

```
python scripts/t2_eta_reservoir.py
python scripts/t2_recal_boundary.py --law project --reps 500 --shard 0 --nshards 2
python scripts/t2_recal_boundary.py --law project --reps 500 --shard 1 --nshards 2
python scripts/t2_analyse.py --law project
```

Run the two shards as separate single-process jobs and merge; a detached
multiprocessing pool loses its result pipe on Windows. Each shard takes about
50 minutes on one core.

Tasks 1, 3 and 4 additionally need `CSDA_PROJECT_ROOT` and the data:

```
python scripts/t1_neff_transport.py
python scripts/t3_link_comparison.py --part pred
python scripts/t3_link_comparison.py --part coef
python scripts/t4_first_passage.py --part matr
python scripts/t4_first_passage.py --part pool
python scripts/t4_analyse.py
python scripts/build_results.py
```

## Seeds

Every seed is recorded and every stochastic step is reproducible.

| Step | Seed |
|---|---|
| Task 1 transport split replay | `20260702 + crc32(key) % 100000` (the published base seed; the replay reproduces the archived splits exactly) |
| Task 2 `eta` reservoir | `20260823` |
| Task 2 recalibration study | `SeedSequence([20260825, cell_index])` |
| Task 2 seed-stability re-run | `SeedSequence([20260926, cell_index])` |
| Task 3 bootstrap | `SeedSequence([20260823, 0])` |
| Task 4 | `20260907`; random-forest comparator keeps `20260531 + L + H` |

## Verification built in

- The Task 1 replay recomputes the published before/after log loss and agrees
  with the archived transport table at all 90 rows to **exactly zero**
  difference, so the reported `n_eff` values describe the very splits behind
  the published transport tables.
- Task 3 reproduces the published cloglog hazard ratio `0.499526` with 95%
  bootstrap interval `(0.471587, 0.524324)` to all six digits, and the
  rolling-origin Brier and AUC ranges to `4.8e-7`, before any link is changed.
- Task 2 was run under a second `eta` law as an implementation reference; it
  reproduces the independent pilot figures to within a factor of 1.3.
- Task 4's two process models are checked against Monte Carlo simulation of the
  processes themselves, and their fitted parameters against the observed median
  endpoint, before any comparison is reported.

Read section 5 of `RESULTS.md` before using any number. It records what was
attempted and rejected, including two numerical defects that had each reversed
a conclusion until the fitted parameters were checked for physical
plausibility.
