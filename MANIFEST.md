# Manuscript to artifact map

Which file backs each numbered table and figure of *Validity Boundaries of
Discrete-Time Hazard Models for Lithium-Ion Battery Reliability*, and which
script produces it.

Read the **Provenance** column carefully. Four of the manuscript's tables come
from the wider project's simulation suite and corpus-harmonisation pipeline,
which are outside this repository; their result tables are included here so
every reported number has a machine-readable source, but the code that
generated them is not part of this package. Everything marked *this repository*
can be regenerated from the scripts here.

## Tables

| Paper | Artifact in `results/` | Produced by | Provenance |
|---|---|---|---|
| Table 1 `tab:link_simulation` | `link_simulation_bias_coverage.csv`, `link_simulation_mean_bias_full_design.csv` | project simulation suite, Study A | upstream |
| Table 2 `tab:censoring_simulation` | `censoring_simulation_bias.csv` | project simulation suite, censoring study | upstream |
| Table 3 `tab:shape_simulation` | `shape_recovery_monotone.csv`, `shape_recovery_misspecified.csv` | project simulation suite, shape study | upstream |
| Table 4 `tab:neff_by_n` | `t2_summary_project.json` (`b_table`), `t2_cells_project.csv` | `scripts/t2_recal_boundary.py`, `scripts/t2_analyse.py` | **this repository** |
| Table 5 `tab:corpus` | `corpus_endpoint_support.csv` | corpus harmonisation pipeline | upstream |
| Table 6 `tab:application_links` | `t3_link_coefficients.csv`, `t3_link_prediction.csv`, `t3_link_penalty_sensitivity.csv` | `scripts/t3_link_comparison.py` | **this repository** |
| Table 7 `tab:first_passage` | `t4_summary.json`, `t4_first_passage_metrics_matr.csv`, `t4_first_passage_params_matr.csv` | `scripts/t4_first_passage.py`, `scripts/t4_analyse.py` | **this repository** |
| Table 8 `tab:transport_neff` | `t1_neff_transport_splits.csv` | `scripts/t1_neff_transport.py` | **this repository** |

## Figures

| Paper | Artifact | Produced by | Provenance |
|---|---|---|---|
| Figure 1 `fig:construction` | `figures/fig1.png` | drawn by hand; not generated from data | schematic |
| Figure 2 `fig:link_discretisation` | `figs/fig_link_grouping.png` = `figures/fig2.png` | `scripts/t_link_figure.py` | **this repository** |
| Figure 3 `fig:neff_spread` | `figs/fig_information_spread.png` = `figures/fig3.png` | `scripts/t2_analyse.py` | **this repository** |
| Figure 4 `fig:firstpassage` | `figs/fig_firstpassage.png` = `figures/fig4.png` | `scripts/t4_analyse.py` | **this repository** |

Two further figures are generated but are not used in the manuscript:
`figs/fig_crossover.png` (the S = 2-4 adaptation boundary) and
`figs/fig_transport_neff.png` (`n_eff` by held-out unit).

## Reference values reproduced before any new analysis

These were checked first, so that the pipelines were known to be faithful to
the published results before anything was changed.

| Quantity | Published | Reproduced |
|---|---|---|
| Standardised lagged-SOH hazard ratio | 0.499526 | to all six digits |
| Its 95% whole-cell bootstrap interval | (0.471587, 0.524324) | to all six digits |
| Rolling-origin Brier range | 0.0366 - 0.0967 | to 4.8e-7 |
| Rolling-origin AUC range | 0.9174 - 0.9949 | to 4.8e-7 |
| Transport before/after log loss, 90 rows | archived transport table | exactly zero difference |
