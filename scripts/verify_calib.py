"""Independent verification of calib.py.

1. rho = 0 fit vs a statsmodels GLM with a cloglog link (unpenalised MLE).
2. penalised fit vs scipy.optimize.minimize on the same objective.
3. n_eff vs the GLM's model-based slope standard error, 1/sqrt(n_eff) == se(b).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402

rng = np.random.default_rng(20260907)
max_ml, max_pen, max_se = 0.0, 0.0, 0.0
cases = 0
for trial in range(40):
    n = int(rng.integers(60, 400))
    m = float(rng.uniform(-3.0, -0.5))
    s = float(rng.uniform(0.4, 1.8))
    b0 = float(rng.uniform(0.6, 2.0))
    eta = m + s * rng.standard_normal(n)
    y = (rng.random(n) < calib.mu_of(0.0 + b0 * eta)).astype(float)
    if y.sum() < 5 or y.sum() > n - 5:
        continue
    cases += 1

    a, b, conv, se = calib.fit_cloglog_recal(eta, y, rho=0.0)
    X = sm.add_constant(eta)
    glm = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.cloglog()))
    res = glm.fit(maxiter=500, tol=1e-12)
    max_ml = max(max_ml, abs(a - res.params[0]), abs(b - res.params[1]))
    max_se = max(max_se, abs(se - res.bse[1]) / max(res.bse[1], 1e-12))

    ne = calib.n_eff(eta, a, b)
    max_se = max(max_se, abs(1.0 / np.sqrt(ne) - se) / max(se, 1e-12))

    for rho in (0.01, 0.1, 1.0, 10.0):
        a1, b1, _, _ = calib.fit_cloglog_recal(eta, y, rho=rho)
        obj = lambda q: calib.negloglik(q[0], q[1], eta, y, rho)
        opt = minimize(obj, [a1, b1], method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
        max_pen = max(max_pen, obj([a1, b1]) - opt.fun)

print(f"cases                                        {cases}")
print(f"max |a,b - statsmodels GLM cloglog MLE|      {max_ml:.3e}")
print(f"max relative se(b) discrepancy (GLM, n_eff)  {max_se:.3e}")
print(f"max penalised objective excess over Nelder-Mead {max_pen:.3e}")
ok = max_ml < 1e-6 and max_se < 1e-5 and max_pen < 1e-8
print("SELF-TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
