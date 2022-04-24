"""
Loss functions
"""

from typing import Optional

import numpy as np
import numpy.typing as npt


def mean_absolute_diff(
    vector1: Optional[npt.NDArray[np.float64]],
    vector2: Optional[npt.NDArray[np.float64]],
) -> float:
    """
    Computes the means absolute difference of two vectors.
    If one of the two vectors is None, return infinity.
    """
    if vector1 is None or vector2 is None:
        return float("inf")

    if vector1.shape[0] != vector2.shape[0]:
        raise ValueError("Vectors need to be of same size.")

    return sum(abs(vector1 - vector2)) / vector1.shape[0]
