"""
Useful utility functions.
"""

from typing import List, Union

import numba
import numpy as np
import numpy.typing as npt

PositionalBounds = Union[List[int], npt.NDArray[np.int64]]


def find_isolated_local_maxima(
    array: List[float], min_distance: int
) -> PositionalBounds:
    """
    Finds local maxima of array which are at least min_distance apart.
    If it finds two maxima closer than min_distance, it prioritizes the higher
    one. If they are the same value, it prioritizes the leftmost one.
    """

    if min_distance < 0:
        raise ValueError("min_distance can't be negative")

    array_length = len(array)
    if array_length == 0:
        return []

    max_pos = int(np.argmax(array))

    left_piece_end = max_pos - min_distance
    right_piece_start = max_pos + min_distance + 1

    if left_piece_end > 0:
        left_piece = array[:left_piece_end]
    else:
        left_piece = []

    if array_length - right_piece_start > 0:
        right_piece = array[right_piece_start:]
    else:
        right_piece = []

    left_piece_maxima = find_isolated_local_maxima(left_piece, min_distance)
    right_piece_maxima = find_isolated_local_maxima(right_piece, min_distance)
    right_piece_maxima = list(map(lambda x: x + right_piece_start, right_piece_maxima))

    return left_piece_maxima + [max_pos] + right_piece_maxima


def find_isolated_local_minima(
    array: List[float], min_distance: int
) -> PositionalBounds:
    """
    Finds local minima of array which are at least min_distance apart.
    If it finds two minima closer than min_distance, it prioritizes the higher
    one. If they are the same value, it prioritizes the leftmost one.
    """
    flipped_array = list(map(lambda x: x * -1, array))
    return find_isolated_local_maxima(flipped_array, min_distance)


def compute_periods(bounds: PositionalBounds) -> npt.NDArray[np.float64]:
    """
    Compute periods based on given bounds. It ignores the first segment.
    """
    return np.diff(bounds)


@numba.njit
def stretch(y: npt.ArrayLike, new_length: int) -> npt.NDArray[np.float64]:
    """
    Stretch or shrink an array to the desired length, interpolating values.
    """
    x = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, new_length)

    y_new = np.interp(x_new, x, y)

    return y_new


def make_template(
    arrays: List[npt.NDArray[np.float64]], n_bins: int = 100
) -> npt.NDArray[np.float64]:
    """
    Finds the template of a list of arrays. The arrays can be of varying length
    and their template is their mean after they have been normalized to have
    the same length.
    """
    if n_bins <= 0:
        raise ValueError("n_bins needs to be a positive integer")

    if len(arrays) == 0:
        return np.empty(0)

    phis = [np.linspace(0, 1, len(time)) for time in arrays]

    flattened_phis = [subitem for item in phis for subitem in item]
    flattened_arrays = np.array([subitem for item in arrays for subitem in item])

    # Rebin and make template
    bins = np.linspace(0, 1, num=n_bins + 1).astype(np.float64)
    idxs = np.digitize(flattened_phis, bins)
    positions = [np.where(idxs == i) for i in range(1, n_bins)]
    # Deals with last bin which needs to be closed on the right
    positions.append(np.where(idxs > n_bins - 1))
    template = np.array([flattened_arrays[pos].mean() for pos in positions])
    return template


def remove_bad_bounds(bounds: PositionalBounds, min_period: float) -> PositionalBounds:
    """
    Removes bounds which period (length) doesn't exceed min_period.
    """
    if min_period <= 0:
        raise ValueError("min_period must be non-negative")
    if len(bounds) == 0:
        raise ValueError("Empty bounds list")

    bounds = bounds.copy()
    while True:
        periods = compute_periods(bounds)
        bad_pos = np.where(periods < min_period)[0] + 1
        bad_pos = _remove_consecutive(bad_pos, stepsize=1)
        bounds = np.delete(bounds, bad_pos)
        if len(bad_pos) == 0:
            break
        print("Removed ", len(bad_pos), "bounds")
    return bounds


def _remove_consecutive(data, stepsize=1):
    if data.shape[0] == 0:
        return data

    consecutive_groups = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return np.array([group[0] for group in consecutive_groups])
