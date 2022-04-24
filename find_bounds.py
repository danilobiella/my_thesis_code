import matplotlib.pyplot as plt
import numpy as np

import src.utils as f
from src.lightcurve import LightCurve, cut_lc, perform_surgery, plot_lc_cycles
from src.loss_functions import mean_absolute_diff
from src.types import PositionalBounds

OBS = "1232"
BOUNDS_FILE = "bounds_refactor2.txt"

BOUNDS_DIR = "interim/bounds"
DATA_DIR = "raw"

DIR = f"/home/danilo/Projects/tesi/data/{OBS}"

CUT_POSITIONS = [(4650, 7850), (10520, 13730), (16780, 18500)]

MIN_PERIOD = 20  # in seconds

N_ITERATIONS = 10


def load_lightcurve(lc_file_path: str) -> LightCurve:
    data = np.loadtxt(lc_file_path, skiprows=2)
    lc = LightCurve(data[:, 0], data[:, 1])
    lc = perform_surgery(lc, CUT_POSITIONS)
    return lc


def estimate_bounds(lc: LightCurve) -> PositionalBounds:
    min_distance = int(60 / 2 / lc.binsize)  # FIXME Parametrize
    bounds = f.find_isolated_local_minima(lc.photon_rate, min_distance)  # type: ignore
    return bounds


def compute_bounds(
    lc: LightCurve, starting_bounds: PositionalBounds
) -> PositionalBounds:
    bounds = starting_bounds

    lc_cycles = cut_lc(lc, bounds)
    template = f.make_template(lc_cycles.photon_rates(), 100)

    for i in range(N_ITERATIONS):
        print("N: ", i, "/", N_ITERATIONS)
        v = 0
        bounds_diff = 6
        while (v < 5) and (bounds_diff > 0):
            print(f"\t{v=}")
            old_bounds = bounds.copy()
            bounds = f.compute_new_bounds(
                bounds,
                template,
                lc.photon_rate,
                loss_function=mean_absolute_diff,
                min_period=int(MIN_PERIOD / lc.binsize),
                delta_space_size=10,
            )
            bounds = np.array(bounds)
            if len(bounds) == len(old_bounds):
                bounds_diff = sum(abs(old_bounds - bounds))
                print("\tbounds diff: ", bounds_diff)
            v += 1

        periods = f.compute_periods(bounds)
        mean_period = np.mean(periods)
        lc_cycles = cut_lc(lc, bounds)
        template = f.make_template(lc_cycles.photon_rates()[:30], int(mean_period))

    return bounds


def main():
    lc = load_lightcurve(f"{DIR}/{DATA_DIR}/lightcurve1A.txt")
    starting_bounds = estimate_bounds(lc)

    lc_cycles = cut_lc(lc, starting_bounds)

    # Plot raw bounds
    fig, ax = plt.subplots(dpi=150, figsize=(10, 4))
    ax = plot_lc_cycles(lc_cycles, ax)
    fig.savefig("plots/starting_bounds.png")

    bounds = compute_bounds(lc, starting_bounds)
    lc_cycles = cut_lc(lc, bounds)

    # Plot final bounds
    fig, ax = plt.subplots(dpi=150, figsize=(10, 4))
    ax = plot_lc_cycles(lc_cycles, ax)
    fig.savefig("plots/final_bounds.png")


if __name__ == "__main__":
    main()
