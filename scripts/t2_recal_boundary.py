"""TASK 2 -- where the boundary between intercept-only updating and
slope-adapting recalibration lies, and whether labelled sample size can locate it.

Design (breaks the confound the reviewer identified: event prevalence and
prediction spread are varied INDEPENDENTLY of the number of labelled units).

    labelled calibration units n : 25, 50, 100, 200, 400
    target event prevalence      : 0.05, 0.20, 0.50
    spread of predictions sd(eta): 0.5, 1.0, 2.0
    true target slope b0         : 0.7, 1.0, 1.25, 1.5, 2.0
    => 225 design cells, 500 replicates each

Arms
    A  intercept-only updating (b = 1)
    B  unrestricted two-parameter ML (rho = 0)
    C  CV-selected penalty, FULL grid {0.01,0.1,1,10,inf} at every n,
       one-standard-error preference for stronger shrinkage, NO staging
    D  the deleted staged rule: n<20 fix b=1; 20<=n<50 grid {0.1,1,10,inf};
       n>=50 full grid.  Reported only as a comparator against C.
    D' the same staged rule with an INDEPENDENT CV fold draw, so that the
       C-vs-D comparison can be reported both with CV noise removed (D, folds
       shared with C) and with it retained (D', as the original code did)
    E  information-adaptive: tau2 = max(0, (b_ML - 1)^2 - 1/n_eff),
       rho = 1/tau2, rho = inf (b = 1) when tau2 = 0

Two target eta laws are run:
    --law project : eta = m + s*Z with Z drawn from the standardised quantile
                    nodes of the project DGP's own fixed-horizon linear
                    predictor (see t2_eta_reservoir.py).  Expected held-out
                    scores are exact sums over the node law.
    --law normal  : eta ~ N(m, s^2), expected scores by 80-point Gauss-Hermite
                    quadrature.  This is the law behind the pilot figures in
                    the task brief and is run only as an implementation check.

Held-out log loss and Brier are computed EXACTLY under the target law, so the
only Monte Carlo variation is the sampling of the calibration set.

Seed 20260825 (the project's recalibration follow-up seed), spawned per cell.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# This study does only 2x2 linear algebra, so threaded BLAS buys nothing and
# its per-thread workspace is what exhausts memory when several worker
# processes are running.  Must be set before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
sys.path.insert(0, str(HERE))

from calib import (  # noqa: E402
    GRID_FULL,
    eb_shrinkage_rho,
    fit_cloglog_recal,
    mu_of,
    n_eff,
    select_rho_cv,
    staged_rho,
)

SEED = 20260825
NS = [25, 50, 100, 200, 400]
PREVS = [0.05, 0.20, 0.50]
SDS = [0.5, 1.0, 2.0]
B0S = [0.7, 1.0, 1.25, 1.5, 2.0]
ARMS = ["A", "B", "C", "D", "Dind", "E"]

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()

_LAW: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def law_nodes(law: str) -> tuple[np.ndarray, np.ndarray]:
    if law not in _LAW:
        if law == "normal":
            _LAW[law] = (_GH_X.copy(), _GH_W.copy())
        elif law == "project":
            z = np.load(QREI / "results" / "t2_eta_nodes.npy")
            _LAW[law] = (z, np.full(z.size, 1.0 / z.size))
        else:
            raise ValueError(law)
    return _LAW[law]


def mean_prev(m, s, a0, b0, z, w):
    return float(np.sum(w * mu_of(a0 + b0 * (m + s * z))))


def solve_m(prev, s, a0, b0, z, w):
    lo, hi = -40.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_prev(mid, s, a0, b0, z, w) < prev:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_scores(a, b, m, s, a0, b0, z, w):
    """Exact expected held-out log loss and Brier under the target law."""
    eta = m + s * z
    pt = np.clip(mu_of(a0 + b0 * eta), 1e-12, 1 - 1e-12)
    ph = np.clip(mu_of(a + b * eta), 1e-6, 1 - 1e-6)
    ll = -np.sum(w * (pt * np.log(ph) + (1 - pt) * np.log(1 - ph)))
    br = np.sum(w * (pt * (1 - ph) ** 2 + (1 - pt) * ph ** 2))
    return float(ll), float(br)


def draw_eta(rng, n, m, s, law, z):
    if law == "normal":
        return m + s * rng.standard_normal(n)
    return m + s * z[rng.integers(0, z.size, n)]


def run_cell(args):
    n, prev, s, b0, reps, seed, law = args
    z, w = law_nodes(law)
    a0 = 0.0
    m = solve_m(prev, s, a0, b0, z, w)
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    ll = {k: [] for k in ARMS}
    br = {k: [] for k in ARMS}
    slope = {k: [] for k in ARMS}
    fixed = {k: 0 for k in ARMS}
    neffs, one_class, same_cd, same_cd_ind = [], 0, 0, 0
    beat_a = {"B": 0, "C": 0, "D": 0, "E": 0}
    conv_fail = 0

    for _ in range(reps):
        eta = draw_eta(rng, n, m, s, law, z)
        p = mu_of(a0 + b0 * eta)
        y = (rng.random(n) < p).astype(float)
        neffs.append(n_eff(eta))
        if y.sum() == 0 or y.sum() == n:
            one_class += 1

        # A -- intercept-only
        a, b, _, _ = fit_cloglog_recal(eta, y, fix_slope=True)
        l, q = expected_scores(a, b, m, s, a0, b0, z, w)
        ll["A"].append(l); br["A"].append(q); slope["A"].append(b); fixed["A"] += 1
        ll_a = l

        # B -- unrestricted ML
        a, b, conv, se_b = fit_cloglog_recal(eta, y, rho=0.0)
        if not conv:
            conv_fail += 1
        l, q = expected_scores(a, b, m, s, a0, b0, z, w)
        ll["B"].append(l); br["B"].append(q); slope["B"].append(b)
        if l < ll_a:
            beat_a["B"] += 1
        b_ml, se_ml = b, se_b

        # shared CV fold assignment for C and D
        folds = min(3, max(2, n // 2))
        perm = rng.permutation(n)

        # C -- unstaged CV ridge, full grid
        r_c = select_rho_cv(eta, y, GRID_FULL, rng, perm=perm)
        a, b, _, _ = fit_cloglog_recal(eta, y, rho=r_c, fix_slope=not np.isfinite(r_c))
        l, q = expected_scores(a, b, m, s, a0, b0, z, w)
        ll["C"].append(l); br["C"].append(q); slope["C"].append(b)
        if not np.isfinite(r_c):
            fixed["C"] += 1
        if l < ll_a:
            beat_a["C"] += 1

        # D -- staged rule, SAME folds as C (isolates the staging itself)
        r_d = staged_rho(n, eta, y, rng, perm=perm)
        a, b, _, _ = fit_cloglog_recal(eta, y, rho=r_d, fix_slope=not np.isfinite(r_d))
        l, q = expected_scores(a, b, m, s, a0, b0, z, w)
        ll["D"].append(l); br["D"].append(q); slope["D"].append(b)
        if not np.isfinite(r_d):
            fixed["D"] += 1
        if l < ll_a:
            beat_a["D"] += 1
        same_cd += int(r_c == r_d)

        # D' -- staged rule with an independent fold draw (CV noise retained)
        r_di = staged_rho(n, eta, y, rng)
        a, b, _, _ = fit_cloglog_recal(eta, y, rho=r_di, fix_slope=not np.isfinite(r_di))
        l, q = expected_scores(a, b, m, s, a0, b0, z, w)
        ll["Dind"].append(l); br["Dind"].append(q); slope["Dind"].append(b)
        if not np.isfinite(r_di):
            fixed["Dind"] += 1
        same_cd_ind += int(r_c == r_di)

        # E -- information-adaptive, no cut points
        r_e, _, _ = eb_shrinkage_rho(eta, y)
        a, b, _, _ = fit_cloglog_recal(eta, y, rho=r_e, fix_slope=not np.isfinite(r_e))
        l, q = expected_scores(a, b, m, s, a0, b0, z, w)
        ll["E"].append(l); br["E"].append(q); slope["E"].append(b)
        if not np.isfinite(r_e):
            fixed["E"] += 1
        if l < ll_a:
            beat_a["E"] += 1

    out = dict(
        law=law, n=n, prev=prev, sd_eta=s, b0=b0, reps=reps, seed=seed,
        m_eta=float(m), folds=int(min(3, max(2, n // 2))),
        n_eff_mean=float(np.mean(neffs)),
        n_eff_median=float(np.median(neffs)),
        n_eff_sd=float(np.std(neffs, ddof=1)),
        n_eff_min=float(np.min(neffs)), n_eff_max=float(np.max(neffs)),
        S=float((b0 - 1.0) ** 2 * np.mean(neffs)),
        one_class_share=one_class / reps,
        same_penalty_CD=same_cd / reps,
        same_penalty_CDind=same_cd_ind / reps,
        ml_nonconvergence_share=conv_fail / reps,
    )
    for k in ARMS:
        out["LL_" + k] = float(np.mean(ll[k]))
        out["LLse_" + k] = float(np.std(ll[k], ddof=1) / np.sqrt(reps))
        out["BR_" + k] = float(np.mean(br[k]))
        out["BRse_" + k] = float(np.std(br[k], ddof=1) / np.sqrt(reps))
        out["sdb_" + k] = float(np.std(slope[k], ddof=1))
        out["fix_share_" + k] = fixed[k] / reps
    for k, v in beat_a.items():
        out["rep_beats_A_" + k] = v / reps
    # paired difference D - C, with its own Monte Carlo SE
    d = np.asarray(ll["D"]) - np.asarray(ll["C"])
    out["LL_D_minus_C"] = float(d.mean())
    out["LL_D_minus_C_se"] = float(d.std(ddof=1) / np.sqrt(reps)) if d.std() > 0 else 0.0
    d2 = np.asarray(ll["Dind"]) - np.asarray(ll["C"])
    out["LL_Dind_minus_C"] = float(d2.mean())
    out["LL_Dind_minus_C_se"] = float(d2.std(ddof=1) / np.sqrt(reps)) if d2.std() > 0 else 0.0
    out["LL_oracle"] = min(out["LL_A"], out["LL_B"])
    return out


def build_cells(reps, law, seed_base, ns=None, prevs=None, sds=None, b0s=None):
    ns = NS if ns is None else ns
    prevs = PREVS if prevs is None else prevs
    sds = SDS if sds is None else sds
    b0s = B0S if b0s is None else b0s
    return [
        (n, p, s, b, reps, [seed_base, i], law)
        for i, (n, p, s, b) in enumerate(itertools.product(ns, prevs, sds, b0s))
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", default="project", choices=["project", "normal"])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: run only the first K cells")
    args = ap.parse_args()

    cells = build_cells(args.reps, args.law, args.seed)
    if args.limit:
        cells = cells[: args.limit]
    tag = args.tag or args.law
    if args.nshards > 1:
        # Sharding across separate single-process jobs, rather than a
        # multiprocessing Pool: a detached Pool on this host loses its result
        # pipe.  The per-cell seed is [seed, cell index] regardless of shard,
        # so the sharded run is identical to an unsharded one.
        cells = cells[args.shard::args.nshards]
        tag = "%s_s%d" % (tag, args.shard)
        print("shard %d/%d: %d cells" % (args.shard, args.nshards, len(cells)), flush=True)
    print("law=%s  %d design cells x %d replicates  seed=%d"
          % (args.law, len(cells), args.reps, args.seed), flush=True)

    dest = QREI / "results"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / ("t2_recal_%s.json" % tag)
    part = dest / ("t2_recal_%s.partial.jsonl" % tag)

    # Results are appended as each design cell finishes, so a run that is
    # interrupted (this machine has 7.3 GB of RAM and the study has been
    # OOM-killed once) can be resumed instead of restarted.
    done = {}
    if args.resume and part.exists():
        for line in part.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done[(r["n"], r["prev"], r["sd_eta"], r["b0"])] = r
        print("resuming: %d cells already complete" % len(done), flush=True)
    todo = [c for c in cells if (c[0], c[1], c[2], c[3]) not in done]

    t0 = time.perf_counter()
    res = list(done.values())
    with open(part, "a", buffering=1) as fh:
        if args.procs <= 1:
            it = (run_cell(c) for c in todo)
        else:
            pool = Pool(args.procs)
            it = pool.imap_unordered(run_cell, todo, chunksize=1)
        for k, r in enumerate(it, 1):
            fh.write(json.dumps(r) + "\n")
            res.append(r)
            el = time.perf_counter() - t0
            print("[%3d/%3d] n=%3d prev=%.2f sd=%.1f b0=%.2f  %.0fs elapsed, "
                  "eta %.0f min" % (k, len(todo), r["n"], r["prev"], r["sd_eta"],
                                    r["b0"], el, (el / k) * (len(todo) - k) / 60.0),
                  flush=True)
        if args.procs > 1:
            pool.close()
            pool.join()
    dt = time.perf_counter() - t0

    path.write_text(json.dumps(res, indent=1))
    print("wrote %s  (%d cells, %.1f s)" % (path, len(res), dt), flush=True)


if __name__ == "__main__":
    main()
