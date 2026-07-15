"""Benchmark dataset loader for the Poker44 pipeline.

Yields (date, sanitized_hands, label) per labeled batch. Sanitization runs every
hand through the validator's own `prepare_hand_for_miner` so training data is
indistinguishable from a live request (train == serve).
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from .payload_view import prepare_hand_for_miner

BENCH_DIR = Path("/root/Skip/poker/SN126/03_data/benchmark/raw")

Batch = Tuple[str, List[dict], int]  # (date, hands, label 1=bot 0=human)


def available_dates() -> List[str]:
    return sorted(
        p.stem.replace("benchmark_", "") for p in BENCH_DIR.glob("benchmark_*.json")
    )


def iter_batches(dates: List[str], *, sanitize: bool = True) -> Iterator[Batch]:
    for date in dates:
        path = BENCH_DIR / f"benchmark_{date}.json"
        with path.open() as f:
            data = json.load(f)["data"]
        for release in data.get("chunks", []):
            groups = release.get("chunks", [])
            labels = release.get("groundTruth", [])
            for hands, label in zip(groups, labels):
                if not isinstance(hands, list) or not hands:
                    continue
                if sanitize:
                    hands = [prepare_hand_for_miner(h) for h in hands]
                yield date, hands, int(label)


def live_size_augment(
    batches: List[Batch],
    *,
    target_min: int = 80,
    target_max: int = 100,
    factor: float = 0.5,
    seed: int = 7,
) -> List[Batch]:
    """Pool same-date, same-label batches into live-sized (80-100 hand) groups.

    Live chunks carry 80-100 hands vs the benchmark's 30-40; without this the
    model never sees the group-size regime it will be scored on.
    """
    rng = random.Random(seed)
    by_key: Dict[Tuple[str, int], List[List[dict]]] = {}
    for date, hands, label in batches:
        by_key.setdefault((date, label), []).append(hands)

    out: List[Batch] = []
    n_target = int(len(batches) * factor)
    keys = list(by_key)
    while len(out) < n_target and keys:
        date, label = keys[rng.randrange(len(keys))]
        pool = by_key[(date, label)]
        if len(pool) < 2:
            continue
        target = rng.randint(target_min, target_max)
        picked: List[dict] = []
        for group in rng.sample(pool, min(len(pool), 4)):
            picked.extend(group)
            if len(picked) >= target:
                break
        if len(picked) < target_min:
            continue
        rng.shuffle(picked)
        out.append((date, picked[:target], label))
    return out


def full_ring_variant(hands: List[dict], seed: int) -> List[dict]:
    """Re-cast a 6-max chunk as a 7-9-max chunk: bump max_seats and re-alias the
    used seats onto a spread of the larger ring, preserving each hand's action
    dynamics and which player is the hero.

    The benchmark is 6-max ONLY, so max_seats/occupancy are constant in training
    and the model cannot learn the seat-count axis — yet 41% of LIVE traffic is
    7-9-max, where every miner (including the leaders) extrapolates. These
    synthetic views expose the model to varied seat counts so its ranking
    degrades less on full-ring live chunks. (It cannot fabricate extra players'
    play, so it teaches seat-count invariance, not new full-ring dynamics.)
    """
    rng = random.Random(seed)
    target = rng.choice([7, 8, 9])
    out: List[dict] = []
    for h in hands:
        h2 = copy.deepcopy(h)
        meta = h2.get("metadata") or {}
        meta["max_seats"] = target
        used = sorted(
            {int(a.get("actor_seat") or 0) for a in (h2.get("actions") or [])}
            | {int(p.get("seat") or 0) for p in (h2.get("players") or [])}
            | {int(meta.get("hero_seat") or 0)}
        )
        used = [s for s in used if s > 0]
        if used and len(used) <= target:
            remap = dict(zip(used, sorted(rng.sample(range(1, target + 1), len(used)))))
            for a in h2.get("actions") or []:
                s = int(a.get("actor_seat") or 0)
                if s in remap:
                    a["actor_seat"] = remap[s]
            for p in h2.get("players") or []:
                s = int(p.get("seat") or 0)
                if s in remap:
                    p["seat"] = remap[s]
            hs = int(meta.get("hero_seat") or 0)
            if hs in remap:
                meta["hero_seat"] = remap[hs]
        h2["metadata"] = meta
        out.append(h2)
    return out
