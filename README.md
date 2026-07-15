# p44_miner_v4 — Poker44 (SN126) bot-detection miner

Model name: **rankblend** (v4). A rank-blended ensemble that scores chunks of
miner-visible poker hands and returns one bot-risk score in `[0, 1]` per chunk.

## Model flow

`chunks -> behavioral features -> 3-member ensemble -> in-batch rank fusion ->
threshold-shaped risk_scores`

- **Features** (`pipeline/features.py`): computed only from miner-visible behavioral
  fields — action-type sequences, big-blind-normalized bet sizings (absolute and
  pot-relative) re-quantized to the validator's bucket grid, action/actor/street
  entropies, pot dynamics, windowing-robust cross-hand signatures, hashed action
  n-grams (sized and size-free), seat-count-invariant per-player rates, and
  **target-player (hero)-conditioned** behavioral features. No hole cards, board cards,
  hand outcomes, timing, or player identifiers. Per-hand features are aggregated to the
  chunk with 7 order statistics.
- **Model** (`pipeline/model.py`): three decorrelated members fused by in-batch rank —
  (1) a regularized stacked GBDT (LightGBM + XGBoost + CatBoost + ExtraTrees →
  LogisticRegression meta), (2) a monotone-constrained LightGBM trio, (3) a
  StandardScaler → PCA → MLP trio.
- **Shaping** (`pipeline/threshold.py`, `serve/scorer.py`): rank fusion sets the
  ordering; a strictly monotone remap moves the deployment threshold onto 0.5 and a
  batch positive budget bounds the positive rate, so AP and recall@FPR reflect the
  model's own ranking.
- **Training** (`pipeline/train.py`, `pipeline/retrain.py`, `pipeline/dataset.py`):
  trained only on the public Poker44 benchmark, sanitized through the validator's
  `prepare_hand_for_miner` (train == serve). Walk-forward validation by release date;
  the shipped model is refit on all dates after passing a no-regression gate; daily
  retrain.

## Serving

`serve/miner.py` is the Bittensor neuron (netuid 126), attaching `serve/scorer.py`
behind the axon and returning `risk_scores` for the `DetectionSynapse` contract.

## Note on weights

Trained model weights are withheld (private). This repository publishes the full
model flow that produces the served risk scores.

License: MIT.
