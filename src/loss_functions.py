"""
Loss functions
"""

from typing import Optional

import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def mean_absolute_diff(
    vector1: Optional[npt.NDArray[np.float64]],
    vector2: Optional[npt.NDArray[np.float64]],
) -> float:
    """
    Computes the means absolute difference of two vectors.
    """
    if vector1.shape[0] != vector2.shape[0]:
        raise ValueError("Vectors need to be of same size.")

    return np.mean(np.abs(vector1 - vector2))
