"""

"""
import matplotlib.pyplot as plt
import numpy as np

import src.lightcurve as lcu
import src.utils as u
from src import bounds_finder
from src.index_functions import first_orbit_cycles

OBS = "1232"
BOUNDS_FILE = "bounds_refactor2.txt"

BOUNDS_DIR = "interim/bounds"
DATA_DIR = "raw"

DIR = f"/home/danilo/Projects/tesi/data/{OBS}"

CUT_POSITIONS = [(4650, 7850), (10520, 13730), (16780, 18500)]

MIN_PERIOD = 20  # in seconds
N_ITERATIONS = 10

LIGHTCURVE_FILE_PATH = f"{DIR}/{DATA_DIR}/lightcurve1A.txt"


def plot_cycles(
    lc: lcu.LightCurve, bounds: u.PositionalBounds, output_filename: str
) -> None:
    """
    Plots
    """
    lc_cycles = lcu.cut_lc(lc, bounds)

    # Plot raw bounds
    fig, ax = plt.subplots(dpi=150, figsize=(10, 4))
    ax = lcu.plot_lc_cycles(lc_cycles, ax)
    fig.savefig(output_filename)


def main():
    """
    Main
    """
    data = np.loadtxt(LIGHTCURVE_FILE_PATH, skiprows=2)
    lc = lcu.LightCurve(data[:, 0], data[:, 1])
    lc = lcu.perform_surgery(lc, CUT_POSITIONS)

    min_distance = int(60 / 2 / lc.binsize)  # FIXME Parametrize
    starting_bounds = u.find_isolated_local_minima(lc.photon_rate, min_distance)  # type: ignore
    plot_cycles(lc, starting_bounds, "plots/starting_bounds.png")

    bounds = bounds_finder.compute_bounds(
        lc,
        starting_bounds,
        index_function=first_orbit_cycles,
        n_iterations=N_ITERATIONS,
        min_period=MIN_PERIOD,
    )
    plot_cycles(lc, bounds, "plots/final_bounds.png")
    # TODO Save bounds times


if __name__ == "__main__":
    main()
