import matplotlib.pyplot as plt
from glob2 import glob
import pandas as pd
import tifffile
import numpy as np
import os
import bactfit
from glob2 import glob
from bactfit.cell import CellList, Cell, ModelCell
from bactfit.preprocess import data_to_cells, mask_to_cells
import pickle


if __name__ == '__main__':

    mask_stack = tifffile.imread("mask_stack.tif") # Load mask stack
    locs = pd.read_csv("localisations.csv") # Load picasso localisations

    cell_dataset = []

    for frame_index, mask in enumerate(mask_stack):
        frame_locs = locs[locs["frame"] == frame_index].copy()

        celllist = mask_to_cells(mask) # Create celllist from binary mask
        celllist.add_localisations(frame_locs) # Add localisations to celllist

        cell_dataset.append(celllist)

    celllist = CellList(cell_dataset) # Create celllist from list of celllists

    celllist.optimise(parallel=True) # Optimise all cells in celllist, fits cell model to cell segmentations

    model = ModelCell(length=12, width=3,)
    celllist.transform_locs(model) # Transform localisations to model cell coordinates

    celllist.plot_heatmap(save=True, path="heatmap.png")
    celllist.plot_render(save=True, path="render.png") #requires picasso metrics