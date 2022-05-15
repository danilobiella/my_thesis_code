import argparse

import matplotlib.pyplot as plt
import numpy as np

import src.lightcurve as lcu


def main(args):
    data = np.loadtxt(args.lc_file, skiprows=2)
    lc = lcu.LightCurve(data[:, 0], data[:, 1])
    lcu.plot_lc(lc)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural net")

    parser.add_argument("--lc_file", required=True, help="Lightcurve input type")

    args = parser.parse_args()
    main(args)
