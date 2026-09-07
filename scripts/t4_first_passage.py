"""TASK 4 -- stochastic-process degradation baselines against the hazard model.

Wiener process with random drift and gamma process with random rate, both with
landmark-conditional first passage to the persistent 80% SOH endpoint
(D* = 0.20 on D(t) = 1 - SOH(t)), compared like-for-like with the existing
complementary log-log hazard model and the existing random-forest comparator.

Two corpora
  matr       the MATR official-endpoint cycle path, 139 cells, with the
             published whole-cell train/test split (97/42).  This is the only
             setting where the existing cloglog and random-forest comparators
             are defined, so it carries the headline Table 4.
  sixsource  the full 493-cell six-source corpus on its own 25-cycle grouped
             representation (D = 1 - lag_soh at each interval start),
             evaluated by five-fold whole-cell cross-validation with
             out-of-fold predictions for every model.  This is where the
             protocol-heterogeneity question the brief raises can be tested.

Fairness requirements
  1. Whole-cell partitioning, identical for every model.
  2. The same nine landmark-horizon pairs, L in {150,200,300}, H in {500,800,1000}.
  3. The same complete-case labelling as R40.make_landmark_table: a cell that
     is event-free at L whose status at H is unknown is EXCLUDED.
  4. Cells that have already crossed the threshold at L are excluded from that
     landmark, for every model.  The published protocol does not do this, so
     `variant=fair` enforces it for all models on an identical cell set and
     `variant=published` reproduces the archived convention.
  In addition, every model is scored on exactly the cells that ALL models can
  score, so no model is advantaged by a different denominator.

Representation
  * Paths live on the project's own 25-cycle observation schedule.
  * The Wiener process is fitted on the RAW series; the gamma process requires
    a non-decreasing path, so a running maximum is applied to D for it.
  * Train window: `full` fits the process parameters on the whole observed
    training path; `truncH` truncates every training path at the horizon H of
    the setting, so no model sees a training observation beyond the prediction
    horizon.  Both are reported.
  * Time scale: `cycle` is raw cycle number.  `normalised` divides cycle
    number by the per-source median observed cycle life.  For a single-source
    corpus that is a pure change of units and the predictions are provably
    invariant (checked numerically); it bites only on the six-source corpus.
    `power` replaces t by t^gamma with gamma estimated by profile likelihood
    and is reported as a DIAGNOSTIC only -- it is an extension beyond the two
    models the brief specifies, included to show whether the failure of the
    first-passage models is the first-passage framework or the assumption
    that degradation increments are time-homogeneous.

Prediction for a test cell uses only its observations at or before L.
Seed 20260907; the random-forest comparator keeps its published per-setting
seed 20260531 + L + H.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
REPRO = QREI / "repro"
os.environ.setdefault("CSDA_PROJECT_ROOT", str(REPRO))
sys.path.insert(0, str(HERE))

import firstpassage as FP  # noqa: E402

LANDMARKS = [150, 200, 300]
HORIZONS = [500, 800, 1000]
THINNING = 25
SEED = 20260907
ALPHA_POOL = 0.03162277660168379
POOL_FEATURES = ["log_interval_mid", "lag_soh", "delta_soh_10"]


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R40 = _import("robustness40", REPRO / "src/pipeline/40_robustness_and_validation_extensions.py")
R41 = _import("domain41", REPRO / "src/pipeline/41_domain_baselines_and_uncertainty.py")
MCS = _import("mcsm79", REPRO / "src/pipeline/79_mcsm_experiment_suite.py")


def metric_row(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    out = dict(
        n_evaluated=int(len(y)), n_events=int(y.sum()),
        observed_rate=float(y.mean()) if len(y) else np.nan,
        mean_pred=float(p.mean()) if len(p) else np.nan,
        brier=float(brier_score_loss(y, p)) if len(y) else np.nan,
        log_loss=float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
        auc=np.nan,
    )
    if len(y) and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def inv_cloglog(eta):
    eta = np.clip(np.asarray(eta, float), -30, 30)
    return np.clip(-np.expm1(-np.exp(eta)), 1e-12, 1 - 1e-12)


# ------------------------------------------------------------------ path build
def matr_paths(cycle_df, timescale, gamma_pow=1.0):
    life = cycle_df.groupby("cell_id").cycle_index.max()
    scale = float(np.median(life)) if timescale == "normalised" else 1.0
    paths = {}
    for cell_id, g0 in cycle_df.groupby("cell_id", sort=False):
        g = g0.sort_values("cycle_index")
        cyc = g.cycle_index.to_numpy(float)
        soh = g.soh.to_numpy(float)
        keep = np.mod(cyc.astype(int), THINNING) == 0
        if keep.sum() < 3:
            keep = np.ones(cyc.size, bool)
        keep[-1] = True
        t, d = cyc[keep] / scale, 1.0 - soh[keep]
        good = np.isfinite(t) & np.isfinite(d) & (t > 0)
        t, d = t[good], d[good]
        if t.size < 3:
            continue
        u = np.concatenate([[True], np.diff(t) > 0])
        t, d = t[u], d[u]
        paths[cell_id] = (t, d, np.maximum.accumulate(d))
    return paths, scale, gamma_pow


def pool_paths(pool, timescale, gamma_pow=1.0):
    """Six-source paths from the grouped representation: t = interval_start,
    D = 1 - lag_soh, i.e. exactly the covariate the hazard model reads."""
    if timescale == "normalised":
        life = pool.groupby(["dataset", "cell_id"]).interval_end.max().reset_index()
        med = life.groupby("dataset").interval_end.median()
        scale_map = med.to_dict()
    else:
        scale_map = None
    paths = {}
    for cell_id, g0 in pool.groupby("cell_id", sort=False):
        g = g0.sort_values("interval_start")
        t = g.interval_start.to_numpy(float)
        d = 1.0 - g.lag_soh.to_numpy(float)
        s = scale_map[str(g.dataset.iloc[0])] if scale_map else 1.0
        t = t / s
        good = np.isfinite(t) & np.isfinite(d) & (t > 0)
        t, d = t[good], d[good]
        if t.size < 3:
            continue
        u = np.concatenate([[True], np.diff(t) > 0])
        t, d = t[u], d[u]
        paths[cell_id] = (t, d, np.maximum.accumulate(d))
    return paths, 1.0, gamma_pow


def warp(paths, gamma_pow):
    if gamma_pow == 1.0:
        return paths
    return {c: (t ** gamma_pow, d, dm) for c, (t, d, dm) in paths.items()}


def truncate(paths, cell_ids, upto, monotone):
    out = []
    for c in cell_ids:
        if c not in paths:
            continue
        t, d, dm = paths[c]
        m = t <= upto
        if m.sum() < 3:
            continue
        out.append((t[m].copy(), (dm if monotone else d)[m].copy()))
    return out


def fit_power(paths, cell_ids, upto):
    """Profile the Wiener likelihood over a power time transform t -> t^gamma."""
    best, bg = None, 1.0
    for gp in [0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        sub = truncate(warp(paths, gp), cell_ids, upto ** gp, monotone=False)
        if len(sub) < 5:
            continue
        try:
            f = FP.fit_wiener(sub)
        except Exception:
            continue
        # compare on a common scale: the Jacobian of the time change does not
        # involve the data, so likelihoods are directly comparable
        if best is None or f["nll"] < best:
            best, bg = f["nll"], gp
    return bg


# ------------------------------------------------------------- landmark tables
_LT_CACHE: dict = {}


def matr_table(cycle_df, splits, landmark, horizon, enforce_req4):
    key = (landmark, horizon, enforce_req4)
    if key in _LT_CACHE:
        return _LT_CACHE[key]
    table = R40.make_landmark_table(cycle_df, splits, landmark, horizon)
    if not table.empty and enforce_req4:
        info = splits.set_index("cell_id")
        obs = info.eol_observed.astype(str).str.lower().eq("true")
        eol = pd.to_numeric(info.eol_cycle, errors="coerce")
        crossed = set(info.index[obs & eol.le(landmark)])
        table = table[~table.cell_id.isin(crossed)].copy()
    _LT_CACHE[key] = table
    return table


def pool_cells(pool, hazard, landmark, horizon, enforce_req4):
    """Complete-case cell-level fixed-horizon table for the six-source pool.

    Returns, per eligible cell, the label and TWO hazard-model predictions:

    `risk_cumulative` accumulates 1 - prod(1 - h_ij) over the intervals in
    [L, H], which is what the published transport analysis does.  It is
    reported but must not be used to rank models: a cell that fails leaves the
    risk set, so it contributes fewer intervals to the window, and the interval
    count alone is almost perfectly inversely predictive of the outcome (see
    section 5).  The artefact is link- and model-dependent, so it makes the
    comparison invalid.

    `lag_soh` / `delta_soh_10` at the last interval at or before L are also
    returned so that a clean landmark-style comparator -- the same kind of
    model the MATR arm uses, fitted directly on the fixed-horizon label with no
    accumulation -- can be fitted out of fold.
    """
    rows = []
    d = pool.assign(h=np.clip(hazard, 1e-12, 1 - 1e-12))
    for cell_id, g0 in d.groupby("cell_id", sort=False):
        g = g0.sort_values("interval_start")
        first = g.iloc[0]
        eol_obs = str(first["eol_observed"]).lower() == "true"
        eol = pd.to_numeric(first["eol_cycle"], errors="coerce")
        cen = pd.to_numeric(first["censor_cycle"], errors="coerce")
        if pd.isna(cen) or cen < landmark:
            continue
        if enforce_req4 and eol_obs and not pd.isna(eol) and eol <= landmark:
            continue
        crossed = eol_obs and not pd.isna(eol) and eol <= horizon
        if not crossed and cen < horizon:
            continue
        win = g[(g.interval_start >= landmark) & (g.interval_start <= horizon)]
        if win.empty:
            continue
        at_l = g[g.interval_start <= landmark]
        if at_l.empty:
            continue
        last = at_l.iloc[-1]
        rows.append(dict(
            cell_id=cell_id, event=int(crossed),
            risk_cumulative=1.0 - float(np.prod(1.0 - win.h.to_numpy())),
            n_intervals=int(len(win)),
            dataset=str(first["dataset"]),
            lag_soh=float(last["lag_soh"]),
            delta_soh_10=float(last["delta_soh_10"]),
        ))
    return pd.DataFrame(rows)


def fit_pool_landmark(train, test):
    """Clean landmark comparator for the pool: cloglog on the fixed-horizon
    label using only features observed at or before L.  No accumulation, so
    the survivorship interval-count artefact cannot enter."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if train.event.nunique() < 2 or test.empty:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.glm("event ~ C(dataset) + lag_soh + delta_soh_10",
                          data=train,
                          family=sm.families.Binomial(
                              link=sm.families.links.CLogLog())).fit(maxiter=200)
        return np.clip(res.predict(test), 1e-9, 1 - 1e-9)
    except Exception:
        return None


# ------------------------------------------------------------------- the runs
def run_matr(timescale, variant, window, use_power=False):
    cycle_df = pd.read_csv(REPRO / "data/processed/matr_official_person_period_cycle.csv")
    splits = pd.read_csv(REPRO / "outputs/audit/matr_official_cell_splits.csv")
    raw_paths, scale, _ = matr_paths(cycle_df, timescale)
    rows, params = [], []
    for landmark in LANDMARKS:
        for horizon in HORIZONS:
            table = matr_table(cycle_df, splits, landmark, horizon,
                               enforce_req4=(variant == "fair"))
            if table.empty:
                continue
            train = table[table.split == "train"].copy()
            test = table[table.split == "test"].copy()

            gp_pow = 1.0
            if use_power:
                gp_pow = fit_power(raw_paths, train.cell_id,
                                   (horizon if window == "truncH" else 1e12) / scale)
            paths = warp(raw_paths, gp_pow)
            L = (landmark / scale) ** gp_pow
            H = (horizon / scale) ** gp_pow
            upto = H if window == "truncH" else 1e18

            wp = FP.fit_wiener(truncate(paths, train.cell_id, upto, monotone=False))
            gg = FP.fit_gamma(truncate(paths, train.cell_id, upto, monotone=True))
            meta = dict(corpus="matr", timescale=timescale, variant=variant,
                        window=window, power=gp_pow, landmark_cycle=landmark,
                        horizon_cycle=horizon)
            params.append(dict(model="wiener", **meta, **wp))
            params.append(dict(model="gamma", **meta, **gg))

            pw, pg = {}, {}
            for c in test.cell_id:
                if c not in paths:
                    continue
                t, d, dm = paths[c]
                a = FP.predict_wiener(t, d, L, H, wp)
                b = FP.predict_gamma(t, dm, L, H, gg)
                if a is not None:
                    pw[c] = a
                if b is not None:
                    pg[c] = b
            common = [c for c in test.cell_id if c in pw and c in pg]
            ev = test[test.cell_id.isin(common)].copy()
            y = ev.event.astype(int).to_numpy()
            base = dict(**meta, n_test_before_scoring=int(len(test)),
                        n_dropped_unscorable=int(len(test) - len(common)))
            rows.append(dict(model="wiener", **base, **metric_row(y, [pw[c] for c in ev.cell_id])))
            rows.append(dict(model="gamma", **base, **metric_row(y, [pg[c] for c in ev.cell_id])))
            ch = R40.fit_predict(train, ev, "event ~ C(batch) + lag_soh + delta_soh_10")
            if ch is not None:
                rows.append(dict(model="cloglog hazard model", **base, **metric_row(y, ch)))
            rf = R41.fit_predict_ml(train, ev, "random_forest",
                                    seed=20260531 + landmark + horizon, calibration="none")
            if rf is not None:
                rows.append(dict(model="random forest", **base, **metric_row(y, rf)))
    return pd.DataFrame(rows), pd.DataFrame(params)


def run_pool(timescale, variant, window):
    pool = MCS.load_pool().reset_index(drop=True)
    raw_paths, _, _ = pool_paths(pool, timescale)
    _, cols = MCS.real_design(pool)

    # out-of-fold hazard predictions, five-fold whole-cell
    oof = np.full(len(pool), np.nan)
    folds = list(GroupKFold(5).split(pool, groups=pool.cell_id))
    for tr, va in folds:
        xtr, _ = MCS.real_design(pool.iloc[tr], cols[1:])
        xva, _ = MCS.real_design(pool.iloc[va], cols[1:])
        fit = MCS.fit_binary(xtr, pool.iloc[tr].event.to_numpy(),
                             link="cloglog", alpha=ALPHA_POOL)
        oof[va] = MCS.inv_link(xva @ fit.beta, "cloglog")

    rows, params = [], []
    for landmark in LANDMARKS:
        for horizon in HORIZONS:
            cell = pool_cells(pool, oof, landmark, horizon,
                              enforce_req4=(variant == "fair"))
            if cell.empty:
                continue
            haz = dict(zip(cell.cell_id, cell.risk_cumulative))
            lab = dict(zip(cell.cell_id, cell.event))
            pw, pg, plm = {}, {}, {}
            for fi, (tr, va) in enumerate(folds):
                train_cells = pool.iloc[tr].cell_id.unique()
                test_cells = [c for c in pool.iloc[va].cell_id.unique() if c in haz]
                if not len(test_cells):
                    continue
                upto = horizon if window == "truncH" else 1e18
                trp_w = truncate(raw_paths, train_cells, upto, monotone=False)
                trp_g = truncate(raw_paths, train_cells, upto, monotone=True)
                if len(trp_w) < 10 or len(trp_g) < 10:
                    continue
                wp = FP.fit_wiener(trp_w)
                gg = FP.fit_gamma(trp_g)
                params.append(dict(model="wiener", corpus="sixsource",
                                   timescale=timescale, variant=variant,
                                   window=window, power=1.0, fold=fi,
                                   landmark_cycle=landmark, horizon_cycle=horizon, **wp))
                params.append(dict(model="gamma", corpus="sixsource",
                                   timescale=timescale, variant=variant,
                                   window=window, power=1.0, fold=fi,
                                   landmark_cycle=landmark, horizon_cycle=horizon, **gg))
                lm = fit_pool_landmark(cell[cell.cell_id.isin(train_cells)],
                                       cell[cell.cell_id.isin(test_cells)])
                if lm is not None:
                    for c, v in zip(cell[cell.cell_id.isin(test_cells)].cell_id,
                                    np.asarray(lm, float)):
                        plm[c] = float(v)
                for c in test_cells:
                    if c not in raw_paths:
                        continue
                    t, d, dm = raw_paths[c]
                    a = FP.predict_wiener(t, d, landmark, horizon, wp)
                    b = FP.predict_gamma(t, dm, landmark, horizon, gg)
                    if a is not None:
                        pw[c] = a
                    if b is not None:
                        pg[c] = b
            common = [c for c in cell.cell_id if c in pw and c in pg and c in plm]
            if not common:
                continue
            y = np.array([lab[c] for c in common], int)
            base = dict(corpus="sixsource", timescale=timescale, variant=variant,
                        window=window, power=1.0, landmark_cycle=landmark,
                        horizon_cycle=horizon,
                        n_test_before_scoring=int(len(cell)),
                        n_dropped_unscorable=int(len(cell) - len(common)))
            rows.append(dict(model="wiener", **base, **metric_row(y, [pw[c] for c in common])))
            rows.append(dict(model="gamma", **base, **metric_row(y, [pg[c] for c in common])))
            rows.append(dict(model="cloglog hazard model", **base,
                             **metric_row(y, [plm[c] for c in common])))
            rows.append(dict(model="cloglog cumulative-risk (artefact-affected)",
                             **base, **metric_row(y, [haz[c] for c in common])))
    return pd.DataFrame(rows), pd.DataFrame(params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "matr", "pool", "selftest"])
    args = ap.parse_args()

    print("first-passage self-test", flush=True)
    if not FP.self_test():
        print("ABORT: reference implementation self-test failed")
        sys.exit(1)
    if args.part == "selftest":
        return

    frames, pframes = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.part in ("all", "matr"):
            for window in ["full", "truncH"]:
                for variant in ["fair", "published"]:
                    for ts in ["cycle", "normalised"]:
                        m, p = run_matr(ts, variant, window)
                        frames.append(m)
                        pframes.append(p)
                        print("  matr %s/%s/%s done" % (window, variant, ts), flush=True)
            # power-transformed diagnostic, headline configuration only
            m, p = run_matr("cycle", "fair", "full", use_power=True)
            m["timescale"] = "power"
            p["timescale"] = "power"
            frames.append(m)
            pframes.append(p)
            print("  matr power-transform diagnostic done", flush=True)
        if args.part in ("all", "pool"):
            for window in ["full", "truncH"]:
                for ts in ["cycle", "normalised"]:
                    m, p = run_pool(ts, "fair", window)
                    frames.append(m)
                    pframes.append(p)
                    print("  sixsource %s/fair/%s done" % (window, ts), flush=True)

    metrics = pd.concat(frames, ignore_index=True)
    params = pd.concat(pframes, ignore_index=True)
    dest = QREI / "results"
    dest.mkdir(parents=True, exist_ok=True)
    tag = "" if args.part == "all" else "_" + args.part
    metrics.to_csv(dest / ("t4_first_passage_metrics%s.csv" % tag), index=False)
    params.to_csv(dest / ("t4_first_passage_params%s.csv" % tag), index=False)
    (dest / "t4_meta.json").write_text(json.dumps(dict(
        seed=SEED, landmarks=LANDMARKS, horizons=HORIZONS,
        thinning=THINNING, threshold_D=FP.THRESH_D,
        pool_alpha=ALPHA_POOL), indent=1))
    print("\nwrote %s (%d rows), params (%d rows)"
          % (dest / ("t4_first_passage_metrics%s.csv" % tag), len(metrics), len(params)))
    for corpus in metrics.corpus.unique():
        sel = metrics[(metrics.corpus == corpus) & (metrics.timescale == "cycle")
                      & (metrics.variant == "fair") & (metrics.window == "full")]
        if sel.empty:
            continue
        print("\n%s, raw cycle scale, requirement-4 enforced, full train window:" % corpus)
        print(sel.groupby("model")[["brier", "log_loss", "auc", "mean_pred",
                                    "observed_rate"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
