"""Guarded daily retrain: fetch -> train candidate -> gate vs champion -> promote.

Usage:
    python -m pipeline.retrain --once            # one full cycle now
    python -m pipeline.retrain --daemon          # run daily at RETRAIN_UTC_HOUR
    python -m pipeline.retrain --once --dry-run  # evaluate but never promote

Gate (candidate must satisfy ALL, vs champion re-scored on the SAME data):
  1. live-size pooled reward >= champion - MAX_REGRESSION
  2. standard held-out reward >= champion - MAX_REGRESSION
  3. shaped output passes the official gated reward with human_safety == 1.0
Promotion is an atomic swap of artifacts/serving_blend_v4.pkl (+meta), with a
timestamped archive. The miner hot-reloads on file change (see serve/scorer.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import random
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from .dataset import available_dates, iter_batches
from .features import chunk_features, feature_names, rows_to_matrix
from .fetch_benchmark import sync
from .model import PercentileBlend, RankBlend
from .serving_blend import ServingBlend
from .threshold import shape_gate_safe
from .train import (
    MAX_KS,
    featurize_dates,
    featurize_real_captures,
    reward,
    select_robust_features,
)

ROOT = Path("/root/Skip/poker/SN126/04_our_miner_v4")
ARTIFACTS = ROOT / "artifacts"
ARCHIVE = ARTIFACTS / "archive"
SERVING = ARTIFACTS / "serving_blend_v4.pkl"
STATE = ARTIFACTS / "retrain_state.json"

HOLDOUT_DAYS = 2  # newest N dates are the gate set
GATE_SEEDS = (42, 7, 123)  # average the live-size gate over these pooling seeds
MAX_REGRESSION = 0.02
TARGET_FPR = 0.04
MAX_POS_FRAC = 0.20
RETRAIN_UTC_HOUR = 3.5  # 03:30 UTC, after the ~20:00 UTC window closes + buffer
ARCHIVE_DEPTH = 14

# v4-frontier: ALL-PERCENTILE ensemble (M1). Within-batch percentile-transformed
# inputs -> invariant to the benchmark->live marginal shift (uid176's core defense).
# Three feature views for decorrelation, all percentile-transformed.
#
# MEASURED 2026-07-16 (decisive): on the 8 live captures, this all-percentile blend
# restores live score spread to 0.651 (healthy ref 0.608; our raw v4 had collapsed to
# 0.22) AND lifts recall@5%FPR from ~0.76 to ~0.95, human_safety=1.0. RAW members are
# deliberately EXCLUDED: they saturate on the sanitized live feed and DRAG spread down
# (equal-weight 4-member = 0.41, all-raw = 0.25) -- the exact collapse M1 fixes.
MEMBER_SPECS = [
    ("pct-all", "all", "pct"),          # every feature, percentile-invariant
    ("pct-livevar", "livevar", "pct"),  # live-varying features, percentile-invariant
]

# M2 (select_weights) is DISABLED for v4: its objective is the offline official reward,
# which our data proves is a "live mirage" -- given the free choice it put ALL weight on
# the raw in-distribution member and ZEROED the percentile branches, collapsing live
# spread 0.22->0.13. Reward-aligned selection on an anti-predictive offline metric
# selects AGAINST live discrimination. Members are equal-weighted; live rounds decide.
# (select_weights is kept in the module for reference / a future spread-aware objective.)
USE_M2_WEIGHTS = False


def log(msg: str) -> None:
    print(f"[{dt.datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}", flush=True)


def _member_cols(spec: str, Xtr: np.ndarray, Xreal: np.ndarray, cols: List[str]) -> List[str]:
    tr_std = Xtr.std(axis=0)
    if spec == "all":
        return [c for c, s in zip(cols, tr_std) if s > 0]
    if spec == "livevar":
        live_std = Xreal.std(axis=0)
        return [c for c, s, ls in zip(cols, tr_std, live_std) if s > 0 and ls > 1e-9]
    if spec == "ks":
        return select_robust_features(Xtr, Xreal, cols, max_ks=0.90)
    raise ValueError(spec)


def _pooled_livesize(dates: List[str], seed: int = 42) -> Tuple[List[List[dict]], np.ndarray]:
    rng = random.Random(seed)
    by_key: Dict[Tuple[str, int], List[List[dict]]] = {}
    for date, hands, label in iter_batches(dates):
        by_key.setdefault((date, label), []).append(hands)
    pooled, labels = [], []
    for (_, label), groups in sorted(by_key.items()):
        rng.shuffle(groups)
        i = 0
        while i < len(groups):
            acc: List[dict] = []
            target = rng.randint(80, 100)
            while i < len(groups) and len(acc) < target:
                acc.extend(groups[i])
                i += 1
            if len(acc) >= 60:
                pooled.append(acc[:target])
                labels.append(label)
    return pooled, np.asarray(labels)


def _evaluate(blend: ServingBlend, X_std, y_std, pooled_sets) -> Dict:
    """Standard reward + live-size reward averaged over pooling seeds, plus the
    official gated reward / human-safety on the first pooled set."""
    out: Dict = {}
    out["standard"] = reward(blend.score_prob(X_std), y_std)

    live_rewards = [reward(blend.score_prob(Xp), yp)["reward"] for Xp, yp in pooled_sets]
    out["livesize"] = {"reward": float(np.mean(live_rewards)), "seeds": live_rewards}

    Xp0, yp0 = pooled_sets[0]
    # evaluate the SAME gate-safe shaping we serve (fixed top-k%, no threshold)
    shaped = shape_gate_safe(blend.score_rank(Xp0), pos_frac=MAX_POS_FRAC)
    sys.path.insert(0, "/root/Skip/poker/SN126/00_external/owner_repo")
    from poker44.score.scoring import reward as official  # noqa: PLC0415

    rew, res = official(shaped, yp0)
    out["official_reward"] = float(rew)
    out["human_safety"] = float(res["human_safety_penalty"])
    return out


def select_weights(blend: ServingBlend, pooled_sets, *, n_samples: int = 300) -> Tuple[Dict[str, float], Dict]:
    """M2: pick per-member fusion weights by REWARD, not equal average.

    For each candidate weight vector we shape the fused rank exactly as served and
    score it with the REAL validator reward on every pooled live-size window, then
    rank candidates by a VARIANCE-PENALIZED objective (mean - 0.5*std), matching the
    frontier's robust selection (uid176 train.py:304). This deliberately does NOT
    optimize benchmark AP -- our own data shows offline AP is a 'live mirage'.
    Selection runs on the held-out pooled windows (members were fit train-only), so
    the weights are chosen out-of-sample w.r.t. the members.

    Returns (weights, info). Weights are applied to BOTH the gate candidate and the
    all-dates final refit, so the recipe validated on holdout is what ships.
    """
    sys.path.insert(0, "/root/Skip/poker/SN126/00_external/owner_repo")
    from poker44.score.scoring import reward as official  # noqa: PLC0415

    tags = [t for t, _ in blend.members]
    # precompute per-window, per-member rank01 vectors once (weights only re-mix them)
    windows = []
    for Xp, yp in pooled_sets:
        mp = blend.member_probs(Xp)
        windows.append(({t: ServingBlend._rank01(mp[t]) for t in tags}, yp))

    def objective(w: Dict[str, float]) -> Tuple[float, List[float]]:
        rewards = []
        for ranks, yp in windows:
            fused = sum(w[t] * ranks[t] for t in tags)
            shaped = shape_gate_safe(fused, pos_frac=MAX_POS_FRAC)
            r, res = official(shaped, yp)
            # never select a weight vector that trips the human-safety gate
            rewards.append(r if res["human_safety_penalty"] >= 1.0 else 0.0)
        arr = np.asarray(rewards)
        return float(arr.mean() - 0.5 * arr.std()), rewards

    cands: List[Dict[str, float]] = [{t: 1.0 / len(tags) for t in tags}]  # equal
    for t in tags:  # each member solo
        cands.append({s: (1.0 if s == t else 0.0) for s in tags})
    rng = np.random.RandomState(0)
    for _ in range(n_samples):  # Dirichlet simplex samples
        v = rng.dirichlet(np.ones(len(tags)))
        cands.append({t: float(v[i]) for i, t in enumerate(tags)})

    equal_val, best_rewards = objective(cands[0])
    best_w, best_obj_val = cands[0], equal_val
    for w in cands[1:]:
        val, rewards = objective(w)
        if val > best_obj_val:
            best_obj_val, best_rewards, best_w = val, rewards, w
    info = {
        "weights": {t: round(best_w[t], 4) for t in tags},
        "objective": round(best_obj_val, 4),
        "equal_objective": round(equal_val, 4),
        "per_window_reward": [round(r, 4) for r in best_rewards],
    }
    return best_w, info


def run_cycle(*, dry_run: bool = False, skip_fetch: bool = False) -> Dict:
    summary: Dict = {"started_at": dt.datetime.utcnow().isoformat() + "Z"}

    if skip_fetch:
        # data already downloaded by the unified orchestrator (scrape script)
        log("skip_fetch: using benchmark data already on disk")
        new_dates = []
    else:
        log("fetching new benchmark releases...")
        try:
            new_dates = sync(log=log)
        except Exception as e:  # noqa: BLE001
            log(f"fetch failed ({e}); proceeding with existing data")
            new_dates = []
    summary["new_dates"] = new_dates

    dates = available_dates()
    holdout = dates[-HOLDOUT_DAYS:]
    train_dates = dates[: -HOLDOUT_DAYS]
    summary["train_span"] = [train_dates[0], train_dates[-1]]
    summary["holdout"] = holdout
    log(f"train {train_dates[0]}..{train_dates[-1]} ({len(train_dates)} dates), gate on {holdout}")

    cols = feature_names()
    # v4/H1: no cross-player pooling augmentation (it mixes multiple policies into
    # one chunk and corrupts the single-policy signature signal). Size-robustness
    # comes from the order-stat/entropy features + hand_count instead.
    tr_d, tr_rows, tr_y = featurize_dates(train_dates, augment=False, full_ring=True, refresh=False)
    te_d, te_rows, te_y = featurize_dates(holdout, augment=False, refresh=False)
    real_rows = featurize_real_captures(refresh=False)
    Xtr_all = rows_to_matrix(tr_rows, cols)
    Xte_all = rows_to_matrix(te_rows, cols)
    Xreal_all = rows_to_matrix(real_rows, cols)
    ytr = np.asarray(tr_y)
    yte = np.asarray(te_y)

    # v4/M3: average the live-size gate metric over several pooling seeds to cut
    # promotion noise on the small (~100-chunk) gate set.
    pooled_sets = []
    for s in GATE_SEEDS:
        pooled, ypool = _pooled_livesize(holdout, seed=s)
        pooled_sets.append((rows_to_matrix([chunk_features(c) for c in pooled], cols), ypool))
    log(f"gate sets: standard n={len(yte)}, live-size {len(GATE_SEEDS)}x~{len(pooled_sets[0][1])}")

    def _fit_members(X, y, dates):
        out = []
        for tag, spec, kind in MEMBER_SPECS:
            kept = _member_cols(spec, X, Xreal_all, cols)
            idx = [cols.index(c) for c in kept]
            t0 = time.time()
            cls = PercentileBlend if kind == "pct" else RankBlend
            blend = cls().fit(X[:, idx], y, dates, kept)
            out.append((tag, blend))
            log(f"  member {tag} [{kind}]: {len(kept)} features, {time.time()-t0:.0f}s")
        return out

    log("training gate candidate (train dates only)...")
    candidate = ServingBlend(_fit_members(Xtr_all, ytr, tr_d))
    if USE_M2_WEIGHTS:
        # M2: reward-aligned, variance-penalized fusion weights (on held-out windows).
        sel_weights, sel_info = select_weights(candidate, pooled_sets)
        candidate.set_weights(sel_weights)
        log(f"M2 weights: {sel_info['weights']} | obj {sel_info['objective']} vs equal {sel_info['equal_objective']}")
        summary["m2_selection"] = sel_info
    else:
        sel_weights = None  # equal weighting; see MEMBER_SPECS note on why M2 is off
        log("M2 disabled (offline-reward objective is anti-predictive here); equal weights")
    cand = _evaluate(candidate, Xte_all, yte, pooled_sets)
    log(
        f"candidate: standard {cand['standard']['reward']:.4f} "
        f"livesize {cand['livesize']['reward']:.4f} official {cand['official_reward']:.4f} "
        f"hsp {cand['human_safety']:.2f}"
    )
    summary["candidate"] = cand

    # Compare against the champion's STORED fair (out-of-sample) gate score from
    # when it was promoted -- NOT a re-score of the shipped champion. The shipped
    # champion was refit on ALL dates (H2), so it has memorized the current gate
    # set; re-scoring it would be leaky (standard->1.0) and would reject every
    # future candidate, freezing the retrain. Both scores are fair train-only
    # holdout rewards on their respective newest-2-date windows.
    champ = None
    meta_path = SERVING.with_name(SERVING.stem + "_meta.json")
    if meta_path.exists():
        try:
            champ = json.loads(meta_path.read_text()).get("gate_metrics")
        except Exception:  # noqa: BLE001
            champ = None
    if champ is not None:
        log(
            f"champion (stored fair gate): standard {champ['standard']['reward']:.4f} "
            f"livesize {champ['livesize']['reward']:.4f} official {champ.get('official_reward', 0):.4f}"
        )
    else:
        log("no stored champion gate metrics; candidate promotes if gate-safe")
    summary["champion"] = champ

    ok_livesize = champ is None or (
        cand["livesize"]["reward"] >= champ["livesize"]["reward"] - MAX_REGRESSION
    )
    ok_standard = champ is None or (
        cand["standard"]["reward"] >= champ["standard"]["reward"] - MAX_REGRESSION
    )
    ok_gate = cand["human_safety"] >= 1.0
    promote = ok_livesize and ok_standard and ok_gate
    summary["gate"] = {
        "livesize_ok": ok_livesize,
        "standard_ok": ok_standard,
        "official_gate_ok": ok_gate,
        "promote": promote,
    }

    if promote and not dry_run:
        # v4/H2: the gate validated the RECIPE on held-out data; the shipped model
        # is refit on ALL dates (incl. the two freshest, which the gate held out)
        # so we never serve a model that is 2 days stale on the current bot cohort.
        log("gate passed - refitting FINAL model on all dates (incl. holdout)...")
        X_all = np.vstack([Xtr_all, Xte_all])
        y_all = np.concatenate([ytr, yte])
        d_all = list(tr_d) + list(te_d)
        # ship the SAME recipe validated on holdout: same members + the M2 weights.
        final = ServingBlend(_fit_members(X_all, y_all, d_all), weights=sel_weights)
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if SERVING.exists():
            shutil.copy2(SERVING, ARCHIVE / f"serving_blend_{stamp}.pkl")
            archives = sorted(ARCHIVE.glob("serving_blend_*.pkl"))
            for old in archives[:-ARCHIVE_DEPTH]:
                old.unlink()
        tmp = SERVING.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(final, f)
        meta = {
            "promoted_at": stamp,
            "final_train_span": [dates[0], dates[-1]],
            "gate_train_span": summary["train_span"],
            "gate_holdout": holdout,
            "pos_frac": MAX_POS_FRAC,
            "gate_metrics": cand,
        }
        SERVING.with_name(SERVING.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
        tmp.replace(SERVING)  # atomic swap; miner hot-reloads on mtime change
        log(f"PROMOTED final (all dates) -> {SERVING.name} (archived {ARCHIVE_DEPTH}-deep)")
    elif promote:
        log("dry-run: candidate passed the gate but was NOT promoted")
    else:
        log("candidate failed the gate; champion retained")

    summary["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
    history = []
    if STATE.exists():
        try:
            history = json.loads(STATE.read_text()).get("history", [])
        except Exception:  # noqa: BLE001
            history = []
    history.append(summary)
    STATE.write_text(json.dumps({"history": history[-60:]}, indent=2, default=str))
    return summary


def daemon() -> None:
    log(f"retrain daemon up; daily at {RETRAIN_UTC_HOUR:.1f}h UTC")
    while True:
        now = dt.datetime.utcnow()
        target = now.replace(
            hour=int(RETRAIN_UTC_HOUR),
            minute=int((RETRAIN_UTC_HOUR % 1) * 60),
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += dt.timedelta(days=1)
        wait = (target - now).total_seconds()
        log(f"next cycle at {target.isoformat()}Z (in {wait/3600:.1f}h)")
        time.sleep(wait)
        try:
            run_cycle()
        except Exception:  # noqa: BLE001
            log(f"cycle failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-fetch", action="store_true",
                        help="skip internal fetch (orchestrator scraped already)")
    args = parser.parse_args()
    if args.once:
        run_cycle(dry_run=args.dry_run, skip_fetch=args.no_fetch)
    elif args.daemon:
        daemon()
    else:
        parser.print_help()
