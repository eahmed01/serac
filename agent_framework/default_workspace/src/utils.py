"""Sample math utilities module.

Provides basic mathematical functions for agent experimentation.
"""

import math


def moving_average(values, window=5):
    """Compute a simple moving average."""
    if len(values) < window:
        return values[:]
    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(sum(values[:i+1]) / (i + 1))
        else:
            result.append(sum(values[i-window+1:i+1]) / window)
    return result


def zscore(value, mean, std):
    """Compute a z-score."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def sigmoid(x):
    """Sigmoid activation function."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def normalize(values):
    """Min-max normalize a list of values to [0, 1]."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]
