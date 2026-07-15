"""Production chunk scorer with a strict reliability contract.

Contract (one violation = the whole evaluation window scores zero):
  1. Always return exactly len(chunks) floats in [0, 1].
  2. Never raise, whatever the payload looks like.
  3. Stay far inside the validator's 180s timeout.

Scoring path: features -> ServingBlend (v1 + live-var + ks0.90) ->
hybrid shaping: ordering from in-request rank fusion (the validator's scoring
window is exactly one request's chunks), positive count from the calibrated
probability blend vs the deploy threshold.
"""
from __future__ import annotations

import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

# Members were fit with sklearn's positional default names (Column_0, ...); we
# predict on numpy arrays built in the exact same column order, so predictions
# are identical (verified: max diff 3e-16). Silence the benign name-mismatch spam.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from pipeline.features import chunk_features
from pipeline.threshold import shape_hybrid

log = logging.getLogger("scorer")

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FALLBACK_SCORE = 0.1  # benign low-risk score for unusable chunks
MAX_HANDS_PER_CHUNK = 120  # runtime cap; live chunks are 80-100


class ChunkScorer:
    def __init__(
        self,
        model_path: Path | str = ARTIFACTS / "serving_blend_v4.pkl",
        *,
        deploy_threshold: float | None = None,
        max_pos_frac: float = 0.16,
    ):
        self.model_path = Path(model_path)
        self.max_pos_frac = float(max_pos_frac)
        self._fixed_threshold = deploy_threshold
        self._mtime = 0.0
        self._load()

    def _load(self) -> None:
        with self.model_path.open("rb") as f:
            self.blend = pickle.load(f)
        self._mtime = self.model_path.stat().st_mtime
        meta_path = self.model_path.with_name(self.model_path.stem + "_meta.json")
        self.deploy_threshold = self._fixed_threshold
        if self.deploy_threshold is None and meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text())
            self.deploy_threshold = float(meta.get("deploy_threshold", 0.5))
        self.deploy_threshold = float(self.deploy_threshold or 0.5)
        log.info(
            "model loaded (mtime=%s, deploy_threshold=%.4f)",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._mtime)),
            self.deploy_threshold,
        )

    def _maybe_reload(self) -> None:
        """Hot-reload when the retrain daemon atomically swaps the artifact."""
        try:
            if self.model_path.stat().st_mtime != self._mtime:
                self._load()
        except Exception:
            log.exception("model hot-reload failed; keeping current model")

    @staticmethod
    def _valid_hand(h: Any) -> bool:
        return isinstance(h, dict) and isinstance(h.get("actions"), list)

    def score_chunks(self, chunks: Sequence[Any]) -> List[float]:
        t0 = time.time()
        n = len(chunks or [])
        if n == 0:
            return []
        self._maybe_reload()

        rows, usable_idx = [], []
        for i, chunk in enumerate(chunks):
            try:
                hands = [h for h in (chunk or []) if self._valid_hand(h)]
                if not hands:
                    continue
                rows.append(chunk_features(hands[:MAX_HANDS_PER_CHUNK]))
                usable_idx.append(i)
            except Exception:
                log.exception("featurization failed for chunk %d", i)

        scores = np.full(n, FALLBACK_SCORE, dtype=float)
        if rows:
            try:
                X = self.blend.featurize(rows)
                shaped = shape_hybrid(
                    self.blend.score_rank(X),
                    self.blend.score_prob(X),
                    deploy_threshold=self.deploy_threshold,
                    max_pos_frac=self.max_pos_frac,
                )
                for j, i in enumerate(usable_idx):
                    scores[i] = shaped[j]
            except Exception:
                log.exception("model scoring failed; serving fallback scores")

        out = [round(float(min(max(s, 0.0), 1.0)), 6) for s in scores]
        log.info(
            "scored %d chunks (%d usable) in %.2fs, positives=%d",
            n, len(usable_idx), time.time() - t0, sum(s >= 0.5 for s in out),
        )
        return out
