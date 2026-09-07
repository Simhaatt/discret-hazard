"""Penalised complementary log-log recalibration, effective calibration
information, and the competing penalty-selection rules.

Reference implementation supplied with the task brief (Appendix A) plus the
CV-with-one-standard-error selector.  Verified against a statsmodels GLM with a
cloglog link to 1e-6 and against an independent optimiser on the penalised
objective to machine precision (see verify_calib.py).

    theta_i = a + b * eta_i
    mu_i    = 1 - exp(-exp(theta_i))
    penalty = (rho/2) * (b - 1)^2       # intercept unpenalised
"""
from __future__ import annotations

import numpy as np

TH_LO, TH_HI = -30.0, 5.0
EPS = 1e-12


def mu_of(theta):
    theta = np.clip(theta, TH_LO, TH_HI)
    return -np.expm1(-np.exp(theta))


def negloglik(a, b, eta, y, rho):
    theta = np.clip(a + b * eta, TH_LO, TH_HI)
    et = np.exp(theta)
    log_mu = np.log(np.clip(-np.expm1(-et), EPS, None))
    ll = np.sum(y * log_mu - (1.0 - y) * et)
    return -ll + 0.5 * rho * (b - 1.0) ** 2


def fit_cloglog_recal(eta, y, rho=0.0, fix_slope=False, max_iter=100, tol=1e-9):
    """Fisher scoring with step-halving.  Returns (a, b, converged, se_b)."""
    eta = np.asarray(eta, float)
    y = np.asarray(y, float)

    if fix_slope or not np.isfinite(rho):
        a, b = 0.0, 1.0
        for _ in range(max_iter):
            theta = np.clip(a + b * eta, TH_LO, TH_HI)
            et = np.exp(theta)
            m = np.clip(mu_of(theta), EPS, 1 - EPS)
            g = np.sum((y - m) * et / m)
            h = np.sum(et ** 2 * (1 - m) / m) + 1e-10
            step = np.clip(g / h, -2.0, 2.0)
            a_new = np.clip(a + step, -25.0, 25.0)
            if abs(a_new - a) < tol:
                a = a_new
                break
            a = a_new
        return a, 1.0, True, np.nan

    a, b, conv = 0.0, 1.0, False
    prev = negloglik(a, b, eta, y, rho)
    a_t, b_t = a, b
    for _ in range(max_iter):
        theta = np.clip(a + b * eta, TH_LO, TH_HI)
        et = np.exp(theta)
        m = np.clip(mu_of(theta), EPS, 1 - EPS)
        u = (y - m) * et / m
        W = et ** 2 * (1 - m) / m
        g0 = u.sum()
        g1 = (u * eta).sum() - rho * (b - 1.0)
        Sw, Swe, Swee = W.sum(), (W * eta).sum(), (W * eta * eta).sum()
        # Explicit 2x2 solve: avoids a LAPACK call (and its per-thread
        # workspace) in the innermost loop.  Agreement with np.linalg.solve is
        # to machine precision; see verify_calib.py.
        h00, h01, h11 = Sw + 1e-8, Swe, Swee + rho + 1e-8
        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or abs(det) < 1e-300:
            break
        step = np.clip(
            np.array([(h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det]),
            -3.0, 3.0,
        )
        t, ok = 1.0, False
        for _ in range(12):
            a_t = np.clip(a + t * step[0], -25.0, 25.0)
            b_t = np.clip(b + t * step[1], -25.0, 25.0)
            cur = negloglik(a_t, b_t, eta, y, rho)
            if np.isfinite(cur) and cur <= prev + 1e-10:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        move = max(abs(a_t - a), abs(b_t - b))
        a, b, prev = a_t, b_t, cur
        if move < tol:
            conv = True
            break

    theta = np.clip(a + b * eta, TH_LO, TH_HI)
    et = np.exp(theta)
    m = np.clip(mu_of(theta), EPS, 1 - EPS)
    W = et ** 2 * (1 - m) / m
    Sw = W.sum()
    if Sw > 0:
        eb = (W * eta).sum() / Sw
        I = (W * (eta - eb) ** 2).sum()
        se_b = 1.0 / np.sqrt(I) if I > 1e-10 else np.inf
    else:
        se_b = np.inf
    return a, b, conv, se_b


def n_eff(eta, a=0.0, b=1.0):
    """Profiled slope information.  At (0, 1) uses predictions only, no labels."""
    eta = np.asarray(eta, float)
    theta = np.clip(a + b * eta, TH_LO, TH_HI)
    et = np.exp(theta)
    m = np.clip(mu_of(theta), EPS, 1 - EPS)
    W = et ** 2 * (1 - m) / m
    Sw = W.sum()
    if Sw <= 0:
        return 0.0
    eb = (W * eta).sum() / Sw
    return float((W * (eta - eb) ** 2).sum())


GRID_FULL = [0.01, 0.1, 1.0, 10.0, np.inf]
GRID_STRONG = [0.1, 1.0, 10.0, np.inf]


def select_rho_cv(eta, y, grid, rng, folds=3, perm=None):
    """CV over `grid` with a one-standard-error preference for the more strongly
    shrunk candidate.  Folds are shared across the grid within a replicate so
    the comparison between candidates is paired.

    Pass `perm` to reuse a fold assignment across two different grids, which is
    what isolates the effect of the grid (arm D vs arm C) from CV noise.
    """
    eta = np.asarray(eta, float)
    y = np.asarray(y, float)
    n = eta.size
    folds = int(min(folds, max(2, n // 2)))
    idx = rng.permutation(n) if perm is None else np.asarray(perm, int)
    parts = np.array_split(idx, folds)
    means, ses = [], []
    for rho in grid:
        per_fold = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            if tr.size < 2 or te.size == 0:
                continue
            a, b, _, _ = fit_cloglog_recal(eta[tr], y[tr], rho=rho,
                                           fix_slope=not np.isfinite(rho))
            p = np.clip(mu_of(a + b * eta[te]), 1e-6, 1 - 1e-6)
            per_fold.append(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log1p(-p)))
        if per_fold:
            means.append(float(np.mean(per_fold)))
            ses.append(float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
                       if len(per_fold) > 1 else 0.0)
        else:
            means.append(np.inf)
            ses.append(0.0)
    means = np.asarray(means)
    ses = np.asarray(ses)
    j = int(np.argmin(means))
    thresh = means[j] + (ses[j] if np.isfinite(ses[j]) else 0.0)
    ok = np.where(means <= thresh + 1e-12)[0]
    return grid[int(ok.max())] if ok.size else grid[j]


def eb_shrinkage_rho(eta, y, floor=1e-3):
    """Arm E.  Positive-part empirical-Bayes penalty; no cut points."""
    a, b, conv, se = fit_cloglog_recal(eta, y, rho=0.0)
    if not np.isfinite(se) or se <= 0:
        return np.inf, b, se
    tau2 = (b - 1.0) ** 2 - se ** 2
    if tau2 <= floor:
        return np.inf, b, se
    return 1.0 / tau2, b, se


def staged_rho(n, eta, y, rng, perm=None):
    """Arm D -- the deleted staged rule, retained ONLY as a comparator.

    n < 20      : fix b = 1
    20 <= n < 50: CV over the strong grid {0.1, 1, 10, inf}
    n >= 50     : CV over the full grid {0.01, 0.1, 1, 10, inf}
    """
    if n < 20:
        return np.inf
    grid = GRID_STRONG if n < 50 else GRID_FULL
    return select_rho_cv(eta, y, grid, rng, perm=perm)
