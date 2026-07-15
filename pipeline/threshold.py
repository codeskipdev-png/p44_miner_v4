"""Threshold shaping: convert ranking quality into validator reward.

The reward has a hard gate at 0.5 (see 01_subnet_understanding/VALIDATOR_ANALYSIS.md):
  - zero reward if no bot chunk is scored >= 0.5;
  - human-safety decays once human FPR@0.5 exceeds 10%.
`shape_scores` applies ONE strictly monotone piecewise-linear map per batch, so
AP and recall@FPR (rank metrics) are untouched; only threshold placement moves.
"""
from __future__ import annotations

import numpy as np


def fit_deploy_threshold(
    human_scores: np.ndarray,
    *,
    target_fpr: float = 0.04,
) -> float:
    """Score value above which only `target_fpr` of validation humans fall."""
    if len(human_scores) == 0:
        return 0.5
    return float(np.quantile(np.asarray(human_scores, float), 1.0 - target_fpr))


def _remap(scores: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear map sending threshold `t` to 0.5."""
    s = np.asarray(scores, dtype=float)
    t = float(min(max(t, 1e-9), 1 - 1e-9))
    lo = 0.5 * s / t
    hi = 0.5 + 0.5 * (s - t) / (1.0 - t)
    return np.clip(np.where(s < t, lo, hi), 0.0, 1.0)


def shape_scores(
    raw: np.ndarray,
    *,
    deploy_threshold: float,
    max_pos_frac: float = 0.16,
) -> np.ndarray:
    """Single monotone remap enforcing all three gate constraints:

    1. scores >= deploy_threshold land at >= 0.5 (model's tuned operating point);
    2. at most floor(n * max_pos_frac) chunks cross 0.5 (FPR-cliff insurance);
    3. at least one chunk crosses 0.5 (zero-positive gate insurance).
    """
    s = np.asarray(raw, dtype=float)
    n = len(s)
    if n == 0:
        return s

    t_eff = float(deploy_threshold)

    budget = max(1, int(np.floor(n * max_pos_frac)))
    if n > budget:
        desc = np.sort(s)[::-1]
        # Midpoint between the budget-th and (budget+1)-th highest raw scores:
        # at most `budget` items sit above it (barring exact ties).
        t_budget = 0.5 * (desc[budget - 1] + desc[budget])
        t_eff = max(t_eff, t_budget)

    s_max = float(s.max())
    if s_max < t_eff:
        t_eff = s_max - 1e-9  # guarantee one positive; gate must never zero us

    out = _remap(s, t_eff)

    # Exact-tie degenerate case (e.g. constant scores): more than `budget`
    # values sit exactly at/above 0.5. Demote surplus ties just below the gate;
    # ties carry no ranking information, so rank metrics are unaffected.
    pos = np.flatnonzero(out >= 0.5)
    if len(pos) > budget:
        order = pos[np.argsort(-out[pos], kind="mergesort")]
        for k, i in enumerate(order[budget:]):
            out[i] = 0.5 - 1e-9 * (k + 1)
    return out


def shape_hybrid(
    rank_scores: np.ndarray,
    prob_scores: np.ndarray,
    *,
    deploy_threshold: float,
    max_pos_frac: float = 0.16,
) -> np.ndarray:
    """Ordering from rank fusion; positive COUNT from calibrated probabilities.

    Rank fusion gives the best in-request ordering (the validator's scoring
    window is exactly one request's chunks), but rank values carry no absolute
    meaning for the 0.5 gate. So: k = how many chunks the calibrated prob blend
    puts above the deploy threshold (clipped to [1, budget]), and the top-k of
    the FUSED ordering are mapped above 0.5, the rest below. Output is a
    monotone function of `rank_scores`, so AP/recall follow the fused ranking.
    """
    r = np.asarray(rank_scores, dtype=float)
    p = np.asarray(prob_scores, dtype=float)
    n = len(r)
    if n == 0:
        return r
    budget = max(1, int(np.floor(n * max_pos_frac)))
    k = int(np.clip((p >= deploy_threshold).sum(), 1, budget))

    order = np.argsort(-r, kind="mergesort")  # stable: ties broken by index
    out = np.empty(n, dtype=float)
    for pos, i in enumerate(order):
        if pos < k:
            out[i] = 0.5 + 0.5 * (k - pos) / (k + 1)
        else:
            out[i] = 0.5 * (n - pos) / (n - k + 1)
    return out
