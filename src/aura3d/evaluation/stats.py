"""Small, dependency-free statistics used by every experiment report."""
from __future__ import annotations

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (valid at k = 0 or k = n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def rule_of_three_upper(n: int) -> float:
    """~95% upper bound on a failure rate after n trials with zero failures."""
    return 3.0 / n if n else float("nan")
