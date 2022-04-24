"""
This module contain the algorithm used to cut SEE PAPER.
"""
import itertools
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

import src.lightcurve as lcu
import src.utils as u
from src.loss_functions import mean_absolute_diff

IndexFunction = Callable[[lcu.LightCurveCycles], list[int]]
LossFunction = Callable[
    [Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]], float
]


def compute_bounds(
    lc: lcu.LightCurve,
    bounds: u.PositionalBounds,
    index_function: IndexFunction,
    min_period: int,
    n_iterations: int,
    n_subiterations: int = 5,
) -> u.PositionalBounds:
    """
    Given a lightcurve and a set of bounds, finds an improved set
    of bounds by minimizing a given loss function acting between the given
    lightcurve and a synthetic lightcurve which is constructed with the template.

    min_period (int, optional): The minimum period for a cycle not to be
        discarded. This is measured in "positional units", that is,
        if a lightcurve is binned at 0.5 s intervals and you want a
        minimum period of 30, then you must provide a value of 30 / 0.5 = 60.
        Defaults to 20.
    """  # FIXME fix docstring

    lc_cycles = lcu.cut_lc(lc, bounds)
    template = u.make_template(lc_cycles.photon_rates(), 100)

    for i in range(n_iterations):
        print(f"Iteration {i} of {n_iterations}")
        for j in range(n_subiterations):
            print(f"\tsubiteration: {j}")
            old_bounds = bounds.copy()
            bounds = compute_new_bounds(
                bounds=bounds,
                template=template,
                lc=lc.photon_rate,
                loss_function=mean_absolute_diff,
                delta_space_size=10,
            )
            bounds = u.remove_bad_bounds(
                bounds, min_period=int(min_period / lc.binsize)
            )
            bounds = np.array(bounds)
            if len(bounds) == len(old_bounds):
                bounds_diff = sum(abs(old_bounds - bounds))
                print("\tbounds diff: ", bounds_diff)
                if bounds_diff > 0:
                    break

        periods = u.compute_periods(bounds)
        mean_period = np.mean(periods)
        lc_cycles = lcu.cut_lc(lc, bounds)
        cycles_idx = index_function(lc_cycles)
        template = u.make_template(
            lc_cycles.photon_rates(idx=cycles_idx), int(mean_period)
        )

    return bounds


def compute_new_bounds(
    bounds: u.PositionalBounds,
    template: npt.NDArray[np.float64],
    lc: npt.NDArray[np.float64],
    loss_function: LossFunction,
    delta_space_size: int = 5,
) -> u.PositionalBounds:
    """
    Given a lightcurve, a template and a set of bounds, finds an improved set
    of bounds by minimizing a given loss function acting between the given
    lightcurve and a synthetic lightcurve which is constructed with the template.

    Args:
        bounds (PositionalBounds): The starting bounds.
        template (npt.NDArray[np.float64]): The template used to construct
            the synthetic lightcurve.
        lc (npt.NDArray[np.float64]): The lightcurve used to fit the synthetic
            lightcurve.
        loss_function (LossFunction): Loss function to minimize.

        delta_space_size (int, optional): Size of neighborhood of every bound
            which will beused to minimize the loss function, measured in
            "positional units" (see above). Defaults to 5.

    Returns:
        PositionalBounds: The improved bounds

    """  # FIXME fix docstring

    for i in range(len(bounds) - 3):
        bounds_slice = bounds[i : i + 4]
        lc_piece = lc[
            bounds_slice[0] : bounds_slice[-1]
        ]  # Selects a 3 cycle piece of lightcurve

        losses = []
        delta_space = np.arange(-delta_space_size, delta_space_size, 1)
        for delta1, delta2 in itertools.product(delta_space, repeat=2):
            synthetic_lc_piece = _make_synthetic_light_curve(
                template, bounds_slice, delta1, delta2
            )
            loss = loss_function(lc_piece, synthetic_lc_piece)
            losses.append(loss)

        delta1, delta2 = _finds_best_delta(losses, delta_space)

        # New bounds
        bounds[i + 1] = bounds[i + 1] + delta1
        bounds[i + 2] = bounds[i + 2] + delta2

    return bounds


def _make_synthetic_light_curve(
    template: npt.NDArray[np.float64],
    bounds_slice: u.PositionalBounds,
    delta1: int,
    delta2: int,
) -> Optional[npt.NDArray[np.float64]]:
    l1 = bounds_slice[1] + delta1 - bounds_slice[0]
    l2 = bounds_slice[2] + delta2 - bounds_slice[1] - delta1
    l3 = bounds_slice[3] - bounds_slice[2] - delta2

    if l1 * l2 * l3 <= 0:
        return None

    stretched_template1 = u.stretch(template, l1)
    stretched_template2 = u.stretch(template, l2)
    stretched_template3 = u.stretch(template, l3)

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
