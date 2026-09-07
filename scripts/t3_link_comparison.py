"""TASK 3 -- logit and probit against complementary log-log on the real corpus.

Two halves, because the two quantities the task brief quotes for the existing
cloglog model come from two different existing protocols:

  (a) COEFFICIENT.  The 493-cell / 16,538-row phase-2 person-period pool,
      Study C protocol of src/pipeline/79_mcsm_experiment_suite.py: penalised
      grouped hazard on standardised log_interval_mid, lag_soh, delta_soh_10
      plus dataset and chemistry dummies, penalty alpha = 0.0316227766 (the
      penalty selected by five-fold whole-cell CV under the cloglog link, held
      fixed across links so that nothing but the link changes), and 1,000
      whole-cell bootstrap resamples, seed 20260823.
      Published cloglog value: HR 0.499526, 95% CI (0.471587, 0.524324).

  (b) PREDICTION.  The leakage-free rolling-origin evaluation that produced the
      published Brier 0.0366-0.0967 and AUC 0.9174-0.9949: the dynamic-simple
      landmark model `event ~ C(batch) + lag_soh + delta_soh_10` on the MATR
      official-endpoint cycle path, whole-cell train/test split, nine
      landmark-horizon settings, complete-case labelling.
      In addition the same nine settings are evaluated on the full 493-cell
      pool under five-fold whole-cell cross-validation ("pool" rows), which is
      the corpus-consistent version of the same comparison.

Every arm is run for cloglog, logit and probit with nothing else changed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
REPRO = QREI / "repro"
os.environ.setdefault("CSDA_PROJECT_ROOT", str(REPRO))

SEED = 20260823
ALPHA = 0.03162277660168379
N_BOOT = 1000
LINKS = ["cloglog", "logit", "probit"]
LANDMARKS = [150, 200, 300]
HORIZONS = [500, 800, 1000]

LINK_OBJ = {
    "cloglog": sm.families.links.CLogLog,
    "logit": sm.families.links.Logit,
    "probit": sm.families.links.Probit,
}


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MCS = _import("mcsm79", REPRO / "src/pipeline/79_mcsm_experiment_suite.py")
R40 = _import("robustness40", REPRO / "src/pipeline/40_robustness_and_validation_extensions.py")


# --------------------------------------------------------------- (a) coefficient
class FastDesign:
    """Rebuild MCS.real_design on a whole-cell bootstrap sample with numpy.

    real_design standardises the three numeric covariates by the mean and
    (ddof=1) standard deviation OF THE SAMPLE, then appends drop-first dummies
    for dataset and chemistry, reindexed to the reference column order.  The
    published loop rebuilds a 16,538-row DataFrame from 493 per-cell slices on
    every one of the 1,000 resamples; this does the same algebra by row
    indexing.  Equivalence is asserted against real_design before use.
    """

    def __init__(self, df, cols):
        self.cols = cols
        self.num = df[["log_interval_mid", "lag_soh", "delta_soh_10"]].to_numpy(float)
        dummies = pd.get_dummies(df[["dataset", "chemistry"]].astype(str),
                                 drop_first=True, dtype=float)
        # reference levels that drop_first removes, needed to check that a
        # bootstrap sample never loses one (which would change the columns)
        self.ref = {c: sorted(df[c].astype(str).unique())[0] for c in ["dataset", "chemistry"]}
        self.dum = dummies.reindex(columns=cols[4:], fill_value=0.0).to_numpy(float)
        self.rows = {c: np.asarray(v) for c, v in df.groupby("cell_id").indices.items()}
        self.event = df.event.to_numpy(float)
        self.cat = {c: df[c].astype(str).to_numpy() for c in ["dataset", "chemistry"]}

    def build(self, sampled_cells):
        idx = np.concatenate([self.rows[c] for c in sampled_cells])
        num = self.num[idx]
        mu = num.mean(axis=0)
        sd = num.std(axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        z = (num - mu) / sd
        x = np.column_stack([np.ones(idx.size), z, self.dum[idx]])
        return x, self.event[idx], idx

    def levels_intact(self, idx):
        return all(self.ref[c] in set(self.cat[c][idx]) for c in self.ref)


def coefficient_arm() -> pd.DataFrame:
    """Study C point fit + whole-cell bootstrap, once per link."""
    df = MCS.load_pool().reset_index(drop=True)
    print("pool rows %d cells %d events %d"
          % (len(df), df.cell_id.nunique(), int(df.event.sum())))
    x, cols = MCS.real_design(df)
    cells = df.cell_id.unique()
    y = df.event.to_numpy()
    fast = FastDesign(df, cols)

    # --- equivalence check against the published construction
    rng0 = np.random.default_rng(np.random.SeedSequence([SEED, 999]))
    worst = 0.0
    for _ in range(5):
        s = rng0.choice(cells, size=len(cells), replace=True)
        bx, by, _ = fast.build(s)
        bd = pd.concat([df[df.cell_id == c] for c in s], ignore_index=True)
        rx, rcols = MCS.real_design(bd, cols[1:])
        assert rcols == cols, "bootstrap sample changed the design columns"
        worst = max(worst, float(np.abs(bx - rx).max()),
                    float(np.abs(by - bd.event.to_numpy(float)).max()))
    print("  fast design vs real_design: max |diff| = %.3e over 5 resamples" % worst)
    assert worst < 1e-10, worst

    out = []
    for link in LINKS:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))
        point = MCS.fit_binary(x, y, link=link, alpha=ALPHA)
        boots = np.empty((N_BOOT, len(cols)))
        dropped = 0
        for b in range(N_BOOT):
            sampled = rng.choice(cells, size=len(cells), replace=True)
            bx, by, idx = fast.build(sampled)
            if not fast.levels_intact(idx):
                dropped += 1
            bf = MCS.fit_binary(bx, by, link=link, alpha=ALPHA)
            boots[b] = bf.beta
            if (b + 1) % 250 == 0:
                print("  %s bootstrap %d/%d" % (link, b + 1, N_BOOT), flush=True)
        if dropped:
            print("  WARNING: %d/%d resamples lost a reference level" % (dropped, N_BOOT))
        for j, (term, coef) in enumerate(zip(cols, point.beta)):
            s = boots[:, j]
            out.append(dict(
                link=link, term=term, beta=float(coef),
                beta_low=float(np.quantile(s, 0.025)),
                beta_high=float(np.quantile(s, 0.975)),
                exp_beta=float(np.exp(coef)),
                exp_beta_low=float(np.exp(np.quantile(s, 0.025))),
                exp_beta_high=float(np.exp(np.quantile(s, 0.975))),
                boot_sd=float(s.std(ddof=1)),
                alpha=ALPHA, n_bootstrap=int(s.size),
                converged=bool(point.success), seed=SEED,
            ))
    return pd.DataFrame(out)


# --------------------------------------------------------------- (b) prediction
def metric_row(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    out = dict(
        n_cells=int(len(y)), n_events=int(y.sum()),
        observed_event_rate=float(y.mean()) if len(y) else np.nan,
        mean_predicted_risk=float(p.mean()) if len(p) else np.nan,
        brier=float(brier_score_loss(y, p)) if len(y) else np.nan,
        log_loss=float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
        auc=np.nan,
    )
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def fit_predict_link(train, test, formula, link):
    if train["event"].nunique() < 2 or test.empty:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.glm(
                formula=formula, data=train,
                family=sm.families.Binomial(link=LINK_OBJ[link]()),
            ).fit(maxiter=200)
        return np.clip(res.predict(test), 1e-9, 1 - 1e-9)
    except Exception as exc:  # pragma: no cover
        print("    fit failed (%s): %s" % (link, exc))
        return None


def rolling_origin_matr() -> pd.DataFrame:
    """The published leakage-free rolling-origin protocol, one run per link."""
    cycle_df = pd.read_csv(REPRO / "data/processed/matr_official_person_period_cycle.csv")
    splits = pd.read_csv(REPRO / "outputs/audit/matr_official_cell_splits.csv")
    formula = "event ~ C(batch) + lag_soh + delta_soh_10"
    rows = []
    for landmark in LANDMARKS:
        for horizon in HORIZONS:
            table = R40.make_landmark_table(cycle_df, splits, landmark, horizon)
            if table.empty:
                continue
            train = table[table["split"] == "train"].copy()
            test = table[table["split"] == "test"].copy()
            y = test["event"].astype(int).to_numpy()
            for link in LINKS:
                pred = fit_predict_link(train, test, formula, link)
                if pred is None:
                    continue
                rows.append(dict(
                    arm="matr_rolling_origin", link=link,
                    landmark_cycle=landmark, horizon_cycle=horizon,
                    **metric_row(y, pred),
                ))
            print("  MATR rolling origin L=%d H=%d done" % (landmark, horizon), flush=True)
    return pd.DataFrame(rows)


def pool_fixed_horizon(df, hazard, landmark, horizon):
    """Cell-level fixed-horizon risk with complete-case labelling, matching the
    convention used by the transport analysis."""
    rows = []
    d = df.assign(h=np.clip(hazard, 1e-12, 1 - 1e-12))
    for cell_id, g0 in d.groupby("cell_id", sort=False):
        g = g0.sort_values("interval_start")
        first = g.iloc[0]
        eol_obs = str(first["eol_observed"]).lower() == "true"
        eol_cycle = pd.to_numeric(first["eol_cycle"], errors="coerce")
        censor = pd.to_numeric(first["censor_cycle"], errors="coerce")
        if pd.isna(censor) or censor < landmark:
            continue
        if eol_obs and not pd.isna(eol_cycle) and eol_cycle <= landmark:
            continue
        crossed = eol_obs and not pd.isna(eol_cycle) and eol_cycle <= horizon
        if not crossed and censor < horizon:
            continue
        win = g[(g["interval_start"] >= landmark) & (g["interval_start"] <= horizon)]
        if win.empty:
            continue
        rows.append(dict(
            cell_id=cell_id, event=int(crossed),
            risk=1.0 - float(np.prod(1.0 - win["h"].to_numpy())),
        ))
    return pd.DataFrame(rows)


def rolling_origin_pool() -> pd.DataFrame:
    """Corpus-consistent version: 493-cell pool, five-fold whole-cell CV, same
    covariates, same penalty, same nine landmark-horizon settings."""
    df = MCS.load_pool().reset_index(drop=True)
    _, cols = MCS.real_design(df)
    oof = {link: np.full(len(df), np.nan) for link in LINKS}
    splitter = GroupKFold(5)
    for fold, (tr, va) in enumerate(splitter.split(df, groups=df.cell_id)):
        train, valid = df.iloc[tr], df.iloc[va]
        xtr, _ = MCS.real_design(train, cols[1:])
        xva, _ = MCS.real_design(valid, cols[1:])
        for link in LINKS:
            fit = MCS.fit_binary(xtr, train.event.to_numpy(), link=link, alpha=ALPHA)
            oof[link][va] = MCS.inv_link(xva @ fit.beta, link)
        print("  pool CV fold %d done" % fold, flush=True)
    rows = []
    for link in LINKS:
        for landmark in LANDMARKS:
            for horizon in HORIZONS:
                cell = pool_fixed_horizon(df, oof[link], landmark, horizon)
                if cell.empty:
                    continue
                rows.append(dict(
                    arm="pool_wholecell_cv", link=link,
                    landmark_cycle=landmark, horizon_cycle=horizon,
                    **metric_row(cell["event"], cell["risk"]),
                ))
    return pd.DataFrame(rows)


def main() -> None:
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "coef", "pred"])
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    N_BOOT = args.boot

    dest = QREI / "results"
    dest.mkdir(parents=True, exist_ok=True)

    if args.part in ("all", "pred"):
        pred = pd.concat([rolling_origin_matr(), rolling_origin_pool()], ignore_index=True)
        pred.to_csv(dest / "t3_link_prediction.csv", index=False)
        print("wrote %s (%d rows)" % (dest / "t3_link_prediction.csv", len(pred)), flush=True)

    if args.part in ("all", "coef"):
        coef = coefficient_arm()
        coef.to_csv(dest / "t3_link_coefficients.csv", index=False)
        print("wrote %s (%d rows)" % (dest / "t3_link_coefficients.csv", len(coef)), flush=True)
        key = coef[coef.term == "lag_soh"]
        print(key[["link", "beta", "beta_low", "beta_high", "exp_beta",
                   "exp_beta_low", "exp_beta_high"]].to_string(index=False))

    (dest / "t3_meta.json").write_text(json.dumps(
        dict(seed=SEED, alpha=ALPHA, n_bootstrap=N_BOOT, links=LINKS,
             landmarks=LANDMARKS, horizons=HORIZONS), indent=1))


if __name__ == "__main__":
    main()
