from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import src.functions as f
from src.functions import PositionalBounds


@dataclass
class LightCurve:
    """"""

    time: npt.NDArray[np.float64]
    photon_rate: npt.NDArray[np.float64]
    binsize: float = field(init=False)

    def __post_init__(self):
        self.binsize = self.time[1] - self.time[0]


@dataclass
class LightCurveCycles:
    cycles: list[LightCurve]

    def __str__(self):
        return f"Number of cycles: {len(self.cycles)}"

    def __iter__(self) -> list[LightCurve]:
        return self.cycles

    def times(self):
        return list(cycle.time for cycle in self.cycles)

    def photon_rates(self) -> list[npt.NDArray[np.float64]]:
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
    """"""
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
    if ax is None:
        _, ax = plt.subplots(dpi=150, figsize=(10, 4))

    for photon_rate in lc_cycles.photon_rates():
        phi = np.linspace(0, 2, len(photon_rate) * 2)
        ax.plot(phi, list(photon_rate) * 2, "-", lw=0.5, ms=1, color="gray", alpha=0.4)
    template = f.make_template(lc_cycles.photon_rates(), 100)

    phi = np.linspace(0, 2, num=200)
    ax.plot(phi, list(template) * 2, color="darkorange")
    return ax


def perform_surgery(
    lc: LightCurve, cut_positions: list[Tuple[float, float]]
) -> LightCurve:
    x2A = []
    y2A = []
    for (a, b) in cut_positions:
        a = int(a / lc.binsize)
        b = int(b / lc.binsize)
        x2A = x2A + list(lc.time[a:b])
        y2A = y2A + list(lc.photon_rate[a:b])

    x2A = np.array(x2A)
    y2A = np.array(y2A)
    return LightCurve(x2A, y2A)
