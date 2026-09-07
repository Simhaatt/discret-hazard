"""TASK 2 verification -- seed stability and Monte Carlo error.

Re-runs a stratified subset of at least 24 design cells with a FRESH seed and
reports the proportion of headline comparisons whose sign is unchanged, plus
the median Monte Carlo standard error per cell.

Headline comparisons checked per cell:
    sign( LL_B - LL_A )   does unrestricted ML beat intercept-only?
    sign( LL_E - LL_A )   does the information-adaptive arm beat it?
    sign( LL_C - LL_A )
    sign( LL_D - LL_C )   does the staging change anything?
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QREI = HERE.parent
sys.path.insert(0, str(HERE))

from t2_recal_boundary import B0S, NS, PREVS, SDS, run_cell  # noqa: E402

# A stratified subset: every n level, every b0 level, and both extremes of
# prevalence and spread.  30 cells.
SELECT = [
    (n, p, s, b)
    for n in NS
    for p in [0.05, 0.50]
    for s in [0.5, 2.0]
    for b in [1.0, 1.5]
] + [(100, 0.20, 1.0, b) for b in [0.7, 1.25, 2.0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", default="project")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260926, help="fresh seed")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    order = {k: i for i, k in enumerate(itertools.product(NS, PREVS, SDS, B0S))}
    cells = [(n, p, s, b, args.reps, [args.seed, order[(n, p, s, b)]], args.law)
             for (n, p, s, b) in SELECT]
    cells = cells[args.shard::args.nshards]
    print("verification: %d cells x %d reps, fresh seed %d (shard %d/%d)"
          % (len(cells), args.reps, args.seed, args.shard, args.nshards), flush=True)

    res = []
    for i, c in enumerate(cells, 1):
        r = run_cell(c)
        res.append(r)
        print("[%2d/%2d] n=%3d prev=%.2f sd=%.1f b0=%.2f" % (i, len(cells), *c[:4]),
              flush=True)

    dest = QREI / "results"
    path = dest / ("t2_verify_%s_s%d.json" % (args.law, args.shard))
    path.write_text(json.dumps(res, indent=1))
    print("wrote %s" % path)


def summarise(law="project", nshards=2) -> dict:
    orig = pd.DataFrame(json.load(open(QREI / "results" / ("t2_recal_%s.json" % law))))
    rep = []
    for s in range(nshards):
        p = QREI / "results" / ("t2_verify_%s_s%d.json" % (law, s))
        if p.exists():
            rep += json.load(open(p))
    rep = pd.DataFrame(rep)
    key = ["n", "prev", "sd_eta", "b0"]
    m = orig.merge(rep, on=key, suffixes=("_o", "_v"))
    comparisons = [("B", "A"), ("E", "A"), ("C", "A"), ("D", "C")]
    rows, agree, total = [], 0, 0
    for hi, lo in comparisons:
        do = m["LL_%s_o" % hi] - m["LL_%s_o" % lo]
        dv = m["LL_%s_v" % hi] - m["LL_%s_v" % lo]
        same = np.sign(do) == np.sign(dv)
        agree += int(same.sum())
        total += int(len(same))
        rows.append(dict(comparison="LL(%s) - LL(%s)" % (hi, lo),
                         cells=int(len(same)),
                         sign_unchanged=int(same.sum()),
                         share=float(same.mean()),
                         max_abs_change=float((do - dv).abs().max())))
    out = dict(law=law, cells_rerun=int(len(m)),
               replicates=int(rep.reps.iloc[0]) if len(rep) else 0,
               fresh_seed_comparisons=total,
               fresh_seed_sign_unchanged=agree,
               fresh_seed_sign_unchanged_share=agree / total if total else np.nan,
               per_comparison=rows,
               median_mcse_per_cell_original=float(
                   orig[["LLse_A", "LLse_B", "LLse_C", "LLse_D", "LLse_E"]].max(axis=1).median()),
               median_mcse_per_cell_rerun=float(
                   rep[["LLse_A", "LLse_B", "LLse_C", "LLse_D", "LLse_E"]].max(axis=1).median())
               if len(rep) else np.nan)
    (QREI / "results" / ("t2_verify_summary_%s.json" % law)).write_text(
        json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    if "--summarise" in sys.argv:
        print(json.dumps(summarise(), indent=1, default=float))
    else:
        main()
