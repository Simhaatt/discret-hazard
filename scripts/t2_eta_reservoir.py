"""TASK 2, stage 0 -- derive the target eta law from the project's own DGP.

The recalibration study needs a distribution for the source model's
fixed-horizon linear predictor eta on target units.  Rather than assume a
normal eta (the assumption behind the pilot figures quoted in the task brief,
Appendix C), this builds the law from the project's Section S1.1
data-generating process:

  1. simulate a SOURCE cohort with `simulate_person_period` (width 25 =>
     J = 40 intervals, smooth increasing log-time baseline, heterogeneous
     declining SOH, analytic state-hazard effect outside the fitted basis,
     one time-varying and one unit-level nuisance covariate, sequential
     first-event generation, independent censoring);
  2. fit the project's penalised grouped complementary log-log hazard on it
     (interval dummies + health + both nuisance covariates, alpha = 0.0316228,
     the manuscript's selected penalty);
  3. simulate an INDEPENDENT target cohort, score it with that fit, and
     accumulate each unit's landmark-to-horizon fixed-horizon risk
     r_i = 1 - prod_j (1 - h_ij) exactly as the battery analysis does;
  4. eta_i = log(-log(1 - r_i)).

The source model is therefore genuinely misspecified for the target (the true
state effect is analytic and outside the fitted basis, and the person-period
observations within a unit are serially dependent) -- unlike the pilot's
correctly specified normal-linear-predictor source.

The standardised quantiles of that eta reservoir define the target law used by
t2_recal_boundary.py.  Because the law is a finite equally-weighted set of
nodes, expected held-out scores are computable exactly by summation over the
nodes, with no evaluation Monte Carlo noise -- the role Gauss-Hermite
quadrature plays for a normal eta.

Seed 20260823 (the project's main simulation seed).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
REPRO = QREI / "repro"
os.environ.setdefault("CSDA_PROJECT_ROOT", str(REPRO))

SEED = 20260823
WIDTH = 25            # => jmax = 1000 / 25 = 40 intervals (project S1.1 design)
EVENT_RATE = 0.02     # mid level of the project's row-event-rate factor
CENSORING = 0.2       # reference design censoring level
N_SOURCE = 1500
N_TARGET = 20000
LANDMARK_J = 6        # cycle 150 with 25-cycle intervals starting after 100
HORIZON_J = 32        # cycle 800
N_NODES = 2000
ALPHA = 0.03162277660168379   # manuscript's selected grouped-hazard penalty


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MCS = _import("mcsm79", REPRO / "src/pipeline/79_mcsm_experiment_suite.py")


def design(df: pd.DataFrame, max_j: int) -> np.ndarray:
    """Grouped-hazard design: interval dummies + nuisance covariates + health."""
    j = df["j"].to_numpy(int)
    baseline = np.eye(max_j)[j - 1, 1:]
    xhealth = np.clip((1.0 - df["soh"].to_numpy()) / 0.2, -0.2, 1.5)
    return np.column_stack(
        [np.ones(len(df)), baseline, df[["ar", "static"]].to_numpy(), xhealth]
    )


def main() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))

    src = MCS.simulate_person_period(rng, N_SOURCE, WIDTH, EVENT_RATE, CENSORING)
    tgt = MCS.simulate_person_period(rng, N_TARGET, WIDTH, EVENT_RATE, CENSORING)
    max_j = int(max(src.j.max(), tgt.j.max()))
    print("source rows %d units %d row-event-rate %.4f" % (len(src), src.cell.nunique(), src.event.mean()))
    print("target rows %d units %d row-event-rate %.4f" % (len(tgt), tgt.cell.nunique(), tgt.event.mean()))
    print("max interval index j = %d" % max_j)

    fit = MCS.fit_binary(design(src, max_j), src.event.to_numpy(), link="cloglog", alpha=ALPHA)
    print("source grouped-hazard fit converged=%s  beta_health=%.4f" % (fit.success, fit.beta[-1]))

    tgt = tgt.copy()
    tgt["h"] = MCS.inv_link(design(tgt, max_j) @ fit.beta, "cloglog")

    # Landmark-conditional fixed-horizon risk, exactly as in the battery
    # analysis: only intervals in (L, H] contribute, and a unit must still be
    # at risk at L.
    window = tgt[(tgt.j > LANDMARK_J) & (tgt.j <= HORIZON_J)]
    at_risk = tgt.groupby("cell").j.max() > LANDMARK_J
    risk = window.groupby("cell").h.apply(lambda p: 1.0 - float(np.prod(1.0 - p.to_numpy())))
    risk = risk[at_risk.reindex(risk.index).fillna(False)]
    risk = risk[(risk > 1e-9) & (risk < 1 - 1e-9)]
    eta = np.log(-np.log1p(-risk.to_numpy(float)))
    print("units contributing eta: %d" % eta.size)
    print("eta  mean %.4f  sd %.4f  min %.4f  max %.4f" % (eta.mean(), eta.std(ddof=1), eta.min(), eta.max()))

    z = (eta - eta.mean()) / eta.std(ddof=1)
    from scipy import stats
    print("standardised eta: skew %.4f  excess kurtosis %.4f" % (stats.skew(z), stats.kurtosis(z)))
    print("Shapiro-Wilk on a 2000-subsample: W=%.4f p=%.3g"
          % stats.shapiro(np.random.default_rng(0).choice(z, 2000, replace=False)))

    # Equally weighted quantile nodes = the target law, re-standardised so the
    # design factor sd(eta) is exact on the node set.
    q = (np.arange(N_NODES) + 0.5) / N_NODES
    nodes = np.quantile(z, q)
    nodes = (nodes - nodes.mean()) / nodes.std(ddof=0)
    print("node law: mean %.3e sd %.6f min %.4f max %.4f"
          % (nodes.mean(), nodes.std(ddof=0), nodes.min(), nodes.max()))

    dest = QREI / "results"
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "t2_eta_nodes.npy", nodes)
    meta = dict(
        seed=SEED, width=WIDTH, intervals=max_j, event_rate=EVENT_RATE,
        censoring=CENSORING, n_source=N_SOURCE, n_target=N_TARGET,
        landmark_j=LANDMARK_J, horizon_j=HORIZON_J, alpha=ALPHA,
        n_nodes=N_NODES, eta_units=int(eta.size),
        eta_mean=float(eta.mean()), eta_sd=float(eta.std(ddof=1)),
        skew=float(stats.skew(z)), excess_kurtosis=float(stats.kurtosis(z)),
        source_fit_converged=bool(fit.success),
        beta_health=float(fit.beta[-1]),
    )
    (dest / "t2_eta_nodes_meta.json").write_text(json.dumps(meta, indent=1))
    print("\nwrote %s and meta" % (dest / "t2_eta_nodes.npy"))


if __name__ == "__main__":
    main()
