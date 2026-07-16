"""ServingBlend: the deployed 3-way model (v1 + live-var + ks0.90).

Rationale (2026-07-15 evaluation, see README):
- labeled live-size proxy (80-100 hand pooled held-out): 0.8821 vs uid227 0.8513
- benchmark held-out: 0.9226 (uid227: 0.9362) - we trade a little in-distribution
  sharpness for live-regime robustness, which is what the validator scores.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .features import feature_names, rows_to_matrix


class ServingBlend:
    def __init__(self, members: List[Tuple[str, object]], weights: Dict[str, float] | None = None):
        self.members = members  # [(tag, RankBlend|PercentileBlend)]
        self.all_cols = feature_names()
        self._col_idx = {
            tag: [self.all_cols.index(c) for c in blend.cols]
            for tag, blend in members
        }
        # M2: per-member fusion weights (reward-aligned selection sets these; default
        # equal == the historical behaviour). Normalized and defensive to missing tags.
        self.weights = self._norm_weights(weights)
        self.meta: Dict = {}

    def _norm_weights(self, weights: Dict[str, float] | None) -> Dict[str, float]:
        tags = [t for t, _ in self.members]
        if not weights:
            return {t: 1.0 / len(tags) for t in tags}
        w = {t: max(float(weights.get(t, 0.0)), 0.0) for t in tags}
        s = sum(w.values())
        return {t: (w[t] / s if s > 0 else 1.0 / len(tags)) for t in tags}

    def _wt(self) -> Dict[str, float]:
        """Backward-compatible weight accessor: models pickled before M2 have no
        `weights` attribute -> fall back to equal, reproducing the old sum/len fusion
        exactly. Prevents a crash when the new code loads an old champion artifact."""
        w = getattr(self, "weights", None)
        if not w:
            return {t: 1.0 / len(self.members) for t, _ in self.members}
        return w

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights = self._norm_weights(weights)

    @staticmethod
    def _rank01(p: np.ndarray) -> np.ndarray:
        if len(p) <= 1:
            return np.full(len(p), 0.5)
        return np.argsort(np.argsort(p, kind="mergesort")) / (len(p) - 1)

    def member_probs(self, X_all: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            tag: blend.score_prob(X_all[:, self._col_idx[tag]])
            for tag, blend in self.members
        }

    def score_prob(self, X_all: np.ndarray) -> np.ndarray:
        w = self._wt()
        probs = self.member_probs(X_all)
        return sum(w[t] * p for t, p in probs.items())

    def score_rank(self, X_all: np.ndarray) -> np.ndarray:
        w = self._wt()
        probs = self.member_probs(X_all)
        return sum(w[t] * self._rank01(p) for t, p in probs.items())

    def featurize(self, rows: List[Dict[str, float]]) -> np.ndarray:
        return rows_to_matrix(rows, self.all_cols)

    @classmethod
    def build(cls, artifact_dir: Path) -> "ServingBlend":
        members = []
        for tag, fname in [
            ("v1", "rankblend_v1.pkl"),
            ("live-var", "sweep_live-var_only.pkl"),
            ("ks0.90", "sweep_ks0.90.pkl"),
        ]:
            with (artifact_dir / fname).open("rb") as f:
                members.append((tag, pickle.load(f)))
        return cls(members)
