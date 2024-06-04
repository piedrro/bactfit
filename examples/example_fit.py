import tifffile
from bactfit.fit import bactfit


if __name__ == "__main__":

    mask = tifffile.imread("mask.tif")

    bf = bactfit()

    bf.fit_cells(mask,
        fit=False,
        parallel=False)

