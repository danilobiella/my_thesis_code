"""
Helper functions
"""

import itertools
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.interpolate

from src.types import LossFunction, PositionalBounds


def find_isolated_local_maxima(
    array: list[float], min_distance: int
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
    array: list[float], min_distance: int
) -> PositionalBounds:
    """
    Finds local minima of array which are at least min_distance apart.
    If it finds two minima closer than min_distance, it prioritizes the higher
    one. If they are the same value, it prioritizes the leftmost one.
    """
    flipped_array = list(map(lambda x: x * -1, array))
    return find_isolated_local_maxima(flipped_array, min_distance)


def compute_periods(
    bounds: PositionalBounds, binsize: Optional[float] = 1.0
) -> npt.NDArray[np.float64]:
    """
    Compute periods based on given bounds. It ignores the first segment.
    """
    periods = np.diff(bounds)
    return periods * binsize


def stretch(y: npt.ArrayLike, new_length: int) -> npt.NDArray[np.float64]:
    """
    stretches a function
    """
    y = np.array(y)
    x = np.linspace(0, 1, num=y.shape[0])

    f = scipy.interpolate.interp1d(x, y)

    x_new = np.linspace(0, 1, num=new_length)
    y_new = f(x_new)

    return y_new


def make_template(
    arrays: list[npt.NDArray[np.float64]], n_bins: int = 100
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


def compute_new_bounds(
    bounds: PositionalBounds,
    template: npt.NDArray[np.float64],
    lc: npt.NDArray[np.float64],
    loss_function: LossFunction,
    min_period: int = 20,
    delta_space_size: int = 5,
) -> PositionalBounds:
    """
    summary_

    Args:
        bounds (PositionalBounds): _description_
        template (npt.NDArray[np.float64]): _description_
        lc (npt.NDArray[np.float64]): _description_
        loss_function (LossFunction): _description_
        min_period (int, optional): _description_. Defaults to 20.
        delta_space_size (int, optional): _description_. Defaults to 5.

    Returns:
        PositionalBounds: _description_
    """
    for i in range(len(bounds) - 3):
        bounds_slice = bounds[i : i + 4]
        lc_piece = lc[
            bounds_slice[0] : bounds_slice[-1]
        ]  # Selects a 3 cycle piece of lightcurve

        losses = []
        delta_space = np.arange(-delta_space_size, delta_space_size, 1)
        for delta1, delta2 in itertools.product(delta_space, repeat=2):
            synthetic_lc_piece = make_synthetic_light_curve(
                template, bounds_slice, delta1, delta2
            )
            loss = loss_function(lc_piece, synthetic_lc_piece)
            losses.append(loss)

        delta1, delta2 = _finds_best_delta(losses, delta_space)

        # New bounds
        bounds[i + 1] = bounds[i + 1] + delta1
        bounds[i + 2] = bounds[i + 2] + delta2

    bounds = remove_bad_bounds(bounds, min_period=min_period)

    return bounds


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


def make_synthetic_light_curve(
    template: npt.NDArray[np.float64],
    bounds_slice: PositionalBounds,
    delta1: int,
    delta2: int,
) -> Optional[npt.NDArray[np.float64]]:
    l1 = bounds_slice[1] + delta1 - bounds_slice[0]
    l2 = bounds_slice[2] + delta2 - bounds_slice[1] - delta1
    l3 = bounds_slice[3] - bounds_slice[2] - delta2

    if l1 * l2 * l3 <= 0:
        return None

    stretched_template1 = stretch(template, l1)
    stretched_template2 = stretch(template, l2)
    stretched_template3 = stretch(template, l3)

    return np.concatenate(
        [stretched_template1, stretched_template2, stretched_template3],
        axis=0,
    )


def _finds_best_delta(
    losses: npt.ArrayLike, delta_space: npt.NDArray[np.int64]
) -> tuple[int, int]:

    losses = np.array(losses)

    best_pos = np.where(losses == min(losses))[0][0]

    delta_1_pos = best_pos // len(delta_space)
    delta_2_pos = best_pos % len(delta_space)

    delta1 = delta_space[delta_1_pos]
    delta2 = delta_space[delta_2_pos]

    return delta1, delta2


def mean_absolute_diff(
    vector1: Optional[npt.NDArray[np.float64]],
    vector2: Optional[npt.NDArray[np.float64]],
) -> float:

    if vector1 is None or vector2 is None:
        return float("inf")

    if vector1.shape[0] != vector2.shape[0]:
        raise ValueError("Vectors need to be of same size.")

    return sum(abs(vector1 - vector2)) / vector1.shape[0]
