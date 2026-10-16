"""Small, dependency-free statistics helpers for benchmark reporting."""

from __future__ import annotations

import math
import random
from typing import Sequence

Z_95 = 1.959963984540054


def mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def wilson_interval(rate: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion ``rate`` observed over ``n`` independent units.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves
    sensibly at 0% / 100% with small n (which benchmark sub-groups often have).
    """

    if n <= 0:
        return 0.0, 1.0
    rate = min(1.0, max(0.0, rate))
    z2 = z * z
    denom = 1 + z2 / n
    center = (rate + z2 / (2 * n)) / denom
    half = z * math.sqrt(rate * (1 - rate) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean; seeded so published reports are reproducible."""

    if not values:
        return 0.0, 1.0
    if len(set(values)) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    alpha = (1 - confidence) / 2
    low = means[int(alpha * resamples)]
    high = means[min(resamples - 1, int((1 - alpha) * resamples))]
    return low, high
