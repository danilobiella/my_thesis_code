"""
Functions and Classes to deal with lightcurves.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import src.utils as f
from src.utils import PositionalBounds


@dataclass
class LightCurve:
    """
    Class to store lightcurve data
    """

    time: npt.NDArray[np.float64]
    photon_rate: npt.NDArray[np.float64]
    binsize: float = field(init=False)

    def __post_init__(self):
        self.binsize = self.time[1] - self.time[0]


@dataclass
class LightCurveCycles:
    """
    Class which represents a cutted lightcurve
    """

    cycles: list[LightCurve]

    def times(self) -> list[npt.NDArray[np.float64]]:
        """
        Return a list containing the time values of the cycles
        """
        return list(cycle.time for cycle in self.cycles)

    def photon_rates(self) -> list[npt.NDArray[np.float64]]:
        """
        Return a list containing the photon rate values of the cycles
        """
        return list(cycle.photon_rate for cycle in self.cycles)


def cut_lc(
    lc: LightCurve, bounds: PositionalBounds, max_cycle_len: int = 500
) -> LightCurveCycles:
    """
    Cut lightcurve into cycles.
    Ignores the segments before and after the limits of bounds
    """
    cycles = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < max_cycle_len / lc.binsize:
            lc_cycle = LightCurve(lc.time[a:b], lc.photon_rate[a:b])
            cycles.append(lc_cycle)

    return LightCurveCycles(cycles)


def plot_lc(lc: LightCurve, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """Plots a lightcurve"""
    if ax is None:
        _, ax = plt.subplots(dpi=150, figsize=(10, 4))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Count rate [counts/s]")

    ax.plot(lc.time, lc.photon_rate, lw=0.5, c="k", **kwargs)

    return ax


def plot_lc_cycles(
    lc_cycles: LightCurveCycles,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots a cutted lightcurve"""
    if ax is None:
        _, ax = plt.subplots(dpi=150, figsize=(10, 4))

    for photon_rate in lc_cycles.photon_rates():
        phi = np.linspace(0, 2, len(photon_rate) * 2)
        ax.plot(phi, list(photon_rate) * 2, "-", lw=0.5, ms=1, color="gray", alpha=0.4)
    template = f.make_template(lc_cycles.photon_rates(), 100)

    phi = np.linspace(0, 2, num=200)
    ax.plot(phi, list(template) * 2, color="darkorange")
    return ax


def perform_surgery(lc: LightCurve, intervals: list[Tuple[int, int]]) -> LightCurve:
    """
    Cuts the lightcurve into intervals and glue them together.
    """
    x = []
    y = []
    for (a, b) in intervals:
        a = int(a / lc.binsize)
        b = int(b / lc.binsize)
        x = x + list(lc.time[a:b])
        y = y + list(lc.photon_rate[a:b])

    x = np.array(x)
    y = np.array(y)
    return LightCurve(x, y)
