"""

"""
import argparse

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


def main(args):
    """
    Main
    """
    data = np.loadtxt(args.lc_file, skiprows=2)
    lightcurve = lcu.LightCurve(data[:, 0], data[:, 1])
    lightcurve = lcu.perform_surgery(lightcurve, CUT_POSITIONS)

    fig, ax = plt.subplots(dpi=150, figsize=(10, 4))
    ax = lcu.plot_lc(lightcurve, ax)
    fig.savefig("plots/lc.png")

    min_distance = int(60 / 2 / lightcurve.binsize)  # FIXME Parametrize
    starting_bounds = u.find_isolated_local_minima(lightcurve.photon_rate, min_distance)

    lc_cycles = lcu.cut_lc(lightcurve, starting_bounds)
    if args.template_in is not None:
        template = np.loadtxt(args.template_in)
    else:
        template = u.make_template(lc_cycles.photon_rates(), 100)

    # Plot cycles
    fig, ax = plt.subplots(dpi=150, figsize=(10, 4))
    ax = lcu.plot_lc_cycles(lc_cycles, ax)
    fig.savefig("plots/starting_bounds.png")

    bounds = bounds_finder.compute_bounds(
        lightcurve,
        template,
        starting_bounds,
        index_function=first_orbit_cycles,
        n_iterations=args.niter,
        min_period=MIN_PERIOD,
    )
    lc_cycles = lcu.cut_lc(lightcurve, bounds)

    # Make template
    if args.template_out is not None:
        template = u.make_template(lc_cycles.photon_rates(), 100)
        np.savetxt(args.template_out, template)

    # Plot cycles
    fig, ax = plt.subplots(dpi=150, figsize=(10, 4))
    ax = lcu.plot_lc_cycles(lc_cycles, ax)
    fig.savefig("plots/final_bounds.png")

    # TODO Save bounds times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural net")

    parser.add_argument("--lc_file", required=True, help="Lightcurve input type")
    parser.add_argument("--niter", type=int, default=10, help="Number of iterations")
    parser.add_argument("--template_in", required=False, help="Template output file")
    parser.add_argument("--template_out", required=False, help="Template output file")

    args = parser.parse_args()
    main(args)
