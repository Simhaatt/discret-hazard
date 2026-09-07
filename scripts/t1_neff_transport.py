"""TASK 1 -- effective calibration information n_eff for every transport split.

Replays the *exact* split sequence of the existing fixed-horizon transport
analysis (src/pipeline/build_manuscript_tables.py :: fixed_horizon_transport,
base seed 20260702, 100 target-cell splits per (holdout, landmark, horizon),
recalibration fractions 0.05/0.10/0.20/0.30 consumed in that order from one
generator per cell) and, for the 30% fraction used in manuscript Tables 8-10,
records the calibration information available on the labelled subset.

n_eff is evaluated at the null recalibration (a = 0, b = 1) and therefore uses
only the source model's fixed-horizon predictions -- no outcome labels.

Also recomputes the archived before/after log loss so the replay can be
verified against results/tables/table_D_*.csv.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
REPRO = QREI / "repro"
os.environ.setdefault("CSDA_PROJECT_ROOT", str(REPRO))
sys.path.insert(0, str(HERE))

import calib  # noqa: E402


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BMT = _import("bmt", REPRO / "src/pipeline/build_manuscript_tables.py")

TARGET_FRAC = 0.30


def main() -> None:
    raw = pd.read_csv(BMT.PROCESSED / "phase2_expanded_person_period_grouped_25.csv")
    raw = raw.dropna(subset=["event", "cell_id", "dataset", "chemistry", *BMT.FEATURES]).copy()
    raw["event"] = raw["event"].astype(int)

    rows = []
    for holdout_type, col in [("chemistry", "chemistry"), ("source", "dataset")]:
        for group in sorted(raw[col].astype(str).unique()):
            mask = raw[col].astype(str).eq(group)
            train_raw = raw[~mask].copy()
            test_raw = raw[mask].copy()
            if train_raw["event"].nunique() < 2:
                print("  skip %s/%s: degenerate training events" % (holdout_type, group))
                continue
            train, _ = BMT.prepare_features(train_raw, test_raw)
            beta = BMT.fit_beta(train)
            test = test_raw.copy()
            _, test_std = BMT.prepare_features(train_raw, test_raw)
            test["pred_row"] = BMT.inv_cloglog(BMT.design(test_std) @ beta)

            for landmark in BMT.LANDMARKS:
                for horizon in BMT.HORIZONS:
                    cell = BMT.fixed_horizon_cells(test, landmark, horizon)
                    if cell.empty:
                        continue
                    key = "%s:%s:%d:%d" % (holdout_type, group, landmark, horizon)
                    rng = np.random.default_rng(
                        BMT.BASE_SEED + crc32(key.encode()) % 100000
                    )
                    # Consume the generator in the original order so the 0.30
                    # splits are identical to the published analysis.
                    for frac in BMT.RECAL_FRACS:
                        cells = cell["cell_id"].drop_duplicates().to_numpy()
                        if len(cells) < 4:
                            continue
                        keep = abs(frac - TARGET_FRAC) <= 1e-9
                        reps = []
                        for _rep in range(BMT.N_REPEATS):
                            cal_cells, eval_cells = BMT.split_cells(cells.copy(), frac, rng)
                            if not keep:
                                continue
                            cal = cell[cell["cell_id"].isin(cal_cells)]
                            ev = cell[cell["cell_id"].isin(eval_cells)]
                            if cal.empty or ev.empty:
                                continue
                            eta = cal["eta_before"].to_numpy(float)
                            yc = cal["event"].to_numpy(int)
                            ne = calib.n_eff(eta)
                            before = BMT.metric_row(ev["event"], ev["risk_before"])
                            a, b = BMT.fit_platt_cloglog(cal["eta_before"], cal["event"])
                            if np.isfinite(a) and np.isfinite(b):
                                after_pred = BMT.inv_cloglog(
                                    a + b * ev["eta_before"].to_numpy(float)
                                )
                            else:
                                after_pred = ev["risk_before"].to_numpy(float)
                            after = BMT.metric_row(ev["event"], after_pred)
                            reps.append(
                                {
                                    "labelled_cells": int(len(eta)),
                                    "events_in_subset": int(yc.sum()),
                                    "one_class_subset": int(len(np.unique(yc)) < 2),
                                    "sd_eta": float(np.std(eta, ddof=1)) if len(eta) > 1 else np.nan,
                                    "mean_eta": float(np.mean(eta)),
                                    "n_eff": ne,
                                    "n_test_cells": before["n_cells"],
                                    "log_loss_before": before["log_loss"],
                                    "log_loss_after": after["log_loss"],
                                    "log_loss_gain": before["log_loss"] - after["log_loss"],
                                }
                            )
                        if not keep or not reps:
                            continue
                        rep = pd.DataFrame(reps)
                        rows.append(
                            {
                                "holdout_type": holdout_type,
                                "holdout_group": group,
                                "landmark_cycle": landmark,
                                "horizon_cycle": horizon,
                                "recalibration_fraction": frac,
                                "n_splits": len(rep),
                                "seed": BMT.BASE_SEED,
                                "target_cells_total": int(cell["cell_id"].nunique()),
                                "target_events_total": int(cell["event"].sum()),
                                "labelled_cells": rep["labelled_cells"].mean(),
                                "events_in_subset": rep["events_in_subset"].mean(),
                                "events_in_subset_min": int(rep["events_in_subset"].min()),
                                "events_in_subset_max": int(rep["events_in_subset"].max()),
                                "one_class_share": rep["one_class_subset"].mean(),
                                "sd_eta": rep["sd_eta"].mean(),
                                "mean_eta": rep["mean_eta"].mean(),
                                "n_eff_mean": rep["n_eff"].mean(),
                                "n_eff_median": rep["n_eff"].median(),
                                "n_eff_p05": rep["n_eff"].quantile(0.05),
                                "n_eff_p95": rep["n_eff"].quantile(0.95),
                                "se_slope_at_mean_neff": float(
                                    1.0 / np.sqrt(rep["n_eff"].mean())
                                ),
                                "log_loss_before": rep["log_loss_before"].mean(),
                                "log_loss_after": rep["log_loss_after"].mean(),
                                "log_loss_gain": rep["log_loss_gain"].mean(),
                            }
                        )
                        print(
                            "  %-9s %-16s L=%3d H=%4d cells=%5.1f ev=%5.2f n_eff=%8.2f"
                            % (
                                holdout_type,
                                group,
                                landmark,
                                horizon,
                                rep["labelled_cells"].mean(),
                                rep["events_in_subset"].mean(),
                                rep["n_eff"].mean(),
                            ),
                            flush=True,
                        )

    out = pd.DataFrame(rows)
    dest = QREI / "results"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "t1_neff_transport_splits.csv", index=False)
    print("\nwrote %s  (%d rows)" % (dest / "t1_neff_transport_splits.csv", len(out)))


if __name__ == "__main__":
    main()
