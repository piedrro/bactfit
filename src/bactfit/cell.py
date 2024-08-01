import random
import string
import time
import copy
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Manager, Event
from functools import partial
from scipy.spatial import distance
import os
import pickle
from shapely.geometry import Polygon, LineString, Point
from shapely.strtree import STRtree
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from picasso.render import render
import warnings
from shapely.affinity import translate

from bactfit.fit import bactfit
from bactfit.postprocess import (cell_coordinate_transformation,
    reflect_loc_horizontally, reflect_loc_vertically)
from bactfit.utils import resize_line, get_vertical
import h5py
import cv2
from matplotlib.colors import ListedColormap
from io import BytesIO
from picasso.render import render



class ModelCell(object):

    def __init__(self, length = 10, radius = 5, margin = 1):

        from bactfit.__init__ import __version__ as bactfit_version
        self.bactfit_version = bactfit_version
        self.type = "ModelCell"

        self.cell_polygon = None
        self.cell_midline = None
        self.cell_centerline = None
        self.cell_radius = radius
        self.cell_length = length
        self.margin = margin

        self.create_model_cell()

    def create_model_cell(self):

        x0 = y0 = self.cell_radius + self.margin

        # Define the coordinates of the line
        midline_x_coords = [x0, x0 + self.cell_length]
        midline_y_coords = [y0, y0]
        midline_coords = list(zip(midline_x_coords, midline_y_coords))
        self.cell_midline = LineString(midline_coords)
        self.cell_polygon = self.cell_midline.buffer(self.cell_radius)

        y0 = self.cell_radius + self.margin
        x0 = self.margin
        centerline_x_coords = [x0, x0 + self.cell_length + (self.cell_radius * 2)]
        centerline_y_coords = [y0, y0]
        centerline_coords = list(zip(centerline_x_coords, centerline_y_coords))
        centerline_coords = np.array(centerline_coords)

        self.cell_centerline = LineString(centerline_coords)
        self.cell_centerline = resize_line(self.cell_centerline, 100)

    def get_image(self, dtype = np.uint16):
        
        width = self.cell_length + ((self.cell_radius+ self.margin) * 2)
        height = self.cell_radius*2 + (self.margin * 2)
        
        image = np.zeros((height,width), dtype=np.uint16)
        
        return image
        

        


class Cell(object):

    def __init__(self, cell_data = None):

        from bactfit.__init__ import __version__ as bactfit_version
        self.bactfit_version = bactfit_version
        self.type = "Cell"

        self.cell_polygon = None
        self.cell_centre = None
        self.bbox = None
        self.height = None
        self.cell_radius = None
        self.vertical = None
        self.name = None
        self.transformed = False

        self.data = {}
        self.locs = []

        #fit data
        self.cell_midline = None
        self.cell_centerline = None
        self.cell_poles = None
        self.cell_index = None
        self.polynomial_params = None
        self.fit_error = None
        self.pixel_size = None

        if cell_data is not None:

            if type(cell_data) == dict:

                for key in cell_data.keys():
                    if key == "radius":
                        setattr(self, "cell_radius", cell_data[key])
                    elif key == "length":
                        setattr(self, "cell_length", cell_data[key])
                    else:
                        setattr(self, key, cell_data[key])

            elif type(cell_data) == np.ndarray:

                cell_mask = cell_data

                if len(cell_mask.shape) != 2:
                    print("Cell data must be a 2D numpy array")
                unique_ids = np.unique(cell_mask)
                if len(unique_ids) != 2:
                    print("Cell data must be a binary mask")
                self.import_mask(cell_mask)

            else:
                print("Cell initialisation must be a dictionary or a 2D numpy array")

        if self.name == None:
            self.generate_name()
            
        if hasattr(self, "midline_coords") and self.cell_midline is None:
            self.cell_midline = LineString(self.midline_coords)
            self.cell_poles = [self.cell_midline.coords[0], self.cell_midline.coords[-1]]

        if hasattr(self, "poly_params") and self.polynomial_params is None:
            self.polynomial_params = self.poly_params

        if self.cell_midline is not None and self.cell_radius is not None:
            self.cell_polygon = self.cell_midline.buffer(self.cell_radius)

        if self.vertical is None and self.cell_polygon is not None:
            self.vertical = get_vertical(self.cell_polygon)

    def __getstate__(self):
        # Return the state as a dictionary, omitting non-picklable attributes
        state = self.__dict__.copy()
        # Remove attributes that cannot be pickled
        # state.pop('non_picklable_attribute', None)
        return state

    def generate_name(self):
        self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        
    def import_mask(self, cell_mask):

        try:
            contours, _ = cv2.findContours(cell_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) > 0:
                contour = contours[0]
                contour = contour.squeeze()

                cell_polygon = Polygon(contour)

                centroid = cell_polygon.centroid
                cell_centre = [centroid.x, centroid.y]

                minx, miny, maxx, maxy = cell_polygon.bounds

                bbox = [minx, miny, maxx, maxy]

                h = maxy - miny
                w = maxx - minx

                if h > w:
                    vertical = True
                else:
                    vertical = False

                self.cell_polygon = cell_polygon
                self.cell_centre = cell_centre
                self.bbox = bbox
                self.height = h
                self.width = w
                self.vertical = vertical
                self.frame_index = 0

        except:
            print(traceback.format_exc())
            return None

    def __setstate__(self, state):
        # Restore the state
        self.__dict__.update(state)

    def remove_locs_outside_cell(self, locs=None):

        if locs is not None:

            if isinstance(locs, pd.DataFrame):
                locs = locs.to_records(index=False)

            self.locs = locs

        if self.locs is None:
            return None

        if len(self.locs) == 0:
            self.locs = None
            return None

        filtered_locs = []

        coords = np.stack([self.locs["x"], self.locs["y"]], axis=1)
        points = [Point(coord) for coord in coords]
        spatial_index = STRtree(points)

        possible_points = spatial_index.query(self.cell_polygon)

        polygon_point_indices = []

        for point_index in possible_points:
            point = points[point_index]

            if self.cell_polygon.contains(point):
                polygon_point_indices.append(point_index)

        if len(polygon_point_indices) > 0:
            polygon_locs = self.locs[polygon_point_indices]

            polygon_locs = pd.DataFrame(polygon_locs)
            polygon_locs = polygon_locs.to_records(index=False)

            filtered_locs.append(polygon_locs)

        if len(filtered_locs) > 0:
            filtered_locs = np.hstack(filtered_locs).view(np.recarray).copy()
            self.locs = filtered_locs
        else:
            self.locs = None

    def transform_cells(self, target_cell=None, locs=None,
            method = "angular", shape_measurements=True):

        if locs is not None:
            self.locs = locs

        if target_cell is not None and self.locs is not None:

            cell = cell_coordinate_transformation(cell=self, target_cell=target_cell,
                method=method, shape_measurements=shape_measurements)

            cell_locs = cell.locs

            if cell_locs is not None:
                self.locs = cell_locs
            else:
                self.locs = None

    def optimise(self, refine_fit = True, fit_mode = "directed_hausdorff",
            min_radius = -1, max_radius = -1, max_error = 5, **kwargs):

        try:
            bf = bactfit(cell=self, refine_fit=refine_fit, fit_mode=fit_mode,
                min_radius=min_radius, max_radius=max_radius, max_error=max_error, **kwargs)

            self = bf.fit()

        except:
            print(traceback.format_exc())
            return None

        return self

    def plot(self):

        if self.cell_polygon is not None:
            polygon = self.cell_polygon
        else:
            polygon = None

        if self.cell_midline is not None:
            midline = self.cell_midline
        else:
            midline = None

        if self.locs is not None:
            locs = self.locs
        else:
            locs = None

        if polygon is not None:
            ploygon_coords = np.array(polygon.exterior.coords)
            plt.plot(*ploygon_coords.T, color="black")
        if midline is not None:
            midline_coords = np.array(midline.coords)
            plt.plot(*midline_coords.T, color="red")
        if locs is not None:
            plt.scatter(locs["x"], locs["y"], color="blue")
        plt.show()

    def add_image(self, image, image_name):

        try:

            if type(image) != np.ndarray:
                print("Image is not a numpy array")
                return None
            if len(image.shape) != 2:
                print("Image is not 2D")
                return None

            if self.cell_polygon is not None:

                cropped_image = self.crop_image(image)

                if cropped_image is None:
                    return None

                self.data[image_name] = cropped_image
        except:
            print(traceback.format_exc())

    def crop_image(self, image, buffer=10):

        cropped_image = None

        try:
            image_shape = image.shape

            minx, miny, maxx, maxy = self.cell_polygon.bounds

            minx = int(minx) - buffer
            miny = int(miny) - buffer
            maxx = int(maxx) + buffer
            maxy = int(maxy) + buffer

            if minx < 0:
                minx = 0
            if miny < 0:
                miny = 0
            if maxx > image_shape[1]:
                maxx = image_shape[1]
            if maxy > image_shape[0]:
                maxy = image_shape[0]

            cropped_image = image[miny:maxy, minx:maxx]

            self.crop_bounds = [minx, miny, maxx, maxy]

        except:
            print(traceback.format_exc())

        return cropped_image

    def get_image(self, image_name):

        if image_name not in self.data.keys():
            return None

        image = self.data[image_name]

        return image

    def get_image_polygon(self):

        if self.cell_polygon is None:
            return None
        if self.crop_bounds is None:
            return None

        minx, miny, maxx, maxy = self.crop_bounds

        polygon = self.cell_polygon

        polygon = translate(polygon, xoff=-minx, yoff=-miny)

        return polygon
    
    def get_image_midline(self):
        
        if self.cell_midline is None:
            return None
        if self.crop_bounds is None:
            return None

        minx, miny, maxx, maxy = self.crop_bounds

        midline = self.cell_midline

        midline = translate(midline, xoff=-minx, yoff=-miny)

        return midline
        
    def get_image_mask(self):

        if self.cell_polygon is None:
            return None
        if self.crop_bounds is None:
            return None

        minx, miny, maxx, maxy = self.crop_bounds

        polygon = self.cell_polygon

        polygon = translate(polygon, xoff=-minx, yoff=-miny)

        polygon_coords = np.array(polygon.exterior.coords)
        mask = np.zeros((maxy-miny, maxx-minx))
        mask = cv2.fillPoly(mask, [polygon_coords.astype(int)], 255)
        mask = mask.astype(bool)

        return mask



    

        

class CellList(object):

    def __init__(self, cell_list, verbose=False):

        from bactfit.__init__ import __version__ as bactfit_version
        self.bactfit_version = bactfit_version
        self.type = "CellList"

        if isinstance(cell_list[0], Cell):
            self.data = cell_list

        elif isinstance(cell_list[0], CellList):
            self.data = []
            for cell_list in cell_list:
                self.data.extend(cell_list.data)

        self.cell_names = []
        self.verbose = verbose

        self.assign_cell_indices()
        self.assign_cell_names()

    def assign_cell_indices(self, reindex=False):

        if reindex == False:
            self.cell_indices = [cell.cell_index for cell in self.data
                                 if cell.cell_index is not None]
        else:
            self.cell_indices = []

        if len(self.cell_indices) == 0:

            for i, cell in enumerate(self.data):
                cell.cell_index = i
                self.cell_indices.append(i)

    def assign_cell_names(self, rename=False):

        if rename == False:
            self.cell_names = [cell.name for cell in self.data if cell.name is not None]
        else:
            self.cell_names = []

        if len(self.cell_names) == 0:

            for cell in self.data:
                cell.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                self.cell_names.append(cell.name)

    @staticmethod
    def compute_optimisation(jobs, progress_list):

        optimised_cells = []

        for job_index, job in enumerate(jobs):

            cell, optimisation_args = job

            try:
                cell = cell.optimise(**optimisation_args)

                if cell is not None:
                    optimised_cells.append(cell)
                    progress_list.append(1)

            except:
                print(traceback.format_exc())
                pass

        return optimised_cells

    def optimise(self, refine_fit=True, fit_mode="directed_hausdorff",
            min_radius = -1, max_radius = -1, max_workers=None, progress_callback=None,
            silence_tqdm=False, parallel=True, max_error=5):

        try:

            if len(self.data) == 0:
                return None

            optimisation_args = {"refine_fit": refine_fit, "fit_mode": fit_mode,
                "min_radius": min_radius, "max_radius": max_radius,
                "max_error": max_error}

            jobs = [(cell, optimisation_args) for cell in self.data]

            n_jobs = len(jobs)
            n_optimised = 0

            if isinstance(jobs[0][0], Cell) and parallel is True:
                n_cpus = os.cpu_count()
                if isinstance(max_workers, int):
                    n_cpus = min(n_cpus, max_workers)
                compute_jobs = np.array_split(jobs, n_cpus)
                executor = ProcessPoolExecutor(max_workers=n_cpus)
            else:
                n_cpus = 1
                compute_jobs = [[job] for job in jobs]
                executor = ThreadPoolExecutor(max_workers=n_cpus)

            print(f"Optimising {n_jobs} cells using {n_cpus} CPU(s) cores")

            optimised_cells = []

            with Manager() as manager:

                progress_list = manager.list()

                with executor:

                    futures = [executor.submit(CellList.compute_optimisation,
                        job, progress_list) for job in compute_jobs]

                    self.compute_progress(futures, n_jobs,
                        progress_list, progress_callback,
                        silence_tqdm=silence_tqdm, tqdm_dec="Optimising Cells")

                    for future in as_completed(futures):
                        try:
                            results = future.result()

                            if results is not None:
                                if len(results) > 0:
                                    optimised_cells.extend(results)
                                    n_optimised += len(results)

                        except Exception as e:
                            print(f"Error: {e}")
                            traceback.print_exc()

            optimised_cells = [cell for cell in optimised_cells if cell is not None]
            self.data = optimised_cells

            print(f"Optimised {n_optimised}/{n_jobs} cells")

            if len(self.data) > 0:
                self.assign_cell_indices(reindex=True)
                self.assign_cell_names()

            error_list = [cell.fit_error for cell in self.data]
            error_list = [error for error in error_list if error is not None]
            print(f"Max fit error: {max(error_list)}")

        except:
            print(traceback.format_exc())
            return None

        return self

    def get_cell_polygons(self, ndim=2, flipxy=False):

        polygons = []
        poly_params = []
        cell_poles = []
        midlines = []
        cell_radii = []
        names = []
        frame_indices = []

        for cell in self.data:
            if hasattr(cell, "cell_polygon"):

                try:

                    cell_polygon = cell.cell_polygon
                    cell_radius = cell.cell_radius
                    cell_midline = cell.cell_midline
                    params = cell.polynomial_params
                    poles = cell.cell_poles

                    if hasattr(cell, "frame_index"):
                        frame_index = cell.frame_index
                    else:
                        frame_index = 0

                    if cell_polygon is not None:

                        name = cell.name
                        cell_polygon = cell_polygon.simplify(0.2)
                        seg = np.array(cell_polygon.exterior.coords)

                        midline = resize_line(cell_midline, 6)
                        midline = np.array(midline.coords)

                        seg = seg[1:]

                        if flipxy:
                            seg = np.fliplr(seg)
                            midline = np.fliplr(midline)

                        if ndim == 3:
                            seg = np.vstack([np.ones(len(seg))*frame_index, seg.T]).T
                            midline = np.vstack([np.ones(len(midline))*frame_index, midline.T]).T

                        polygons.append(seg)
                        names.append(name)
                        midlines.append(midline)
                        cell_radii.append(cell_radius)
                        poly_params.append(params)
                        cell_poles.append(poles)
                        frame_indices.append(frame_index)

                except:
                    pass

        data = {"polygons": polygons,
                "midlines": midlines,
                "cell_radii": cell_radii,
                "names": names,
                "poly_params": poly_params,
                "cell_poles": cell_poles,
                "frame_indices": frame_indices,
                }

        return data

    @staticmethod
    def add_localisation_task(cells, locs, remove_outside, progress_list):

        try:

            for cell in cells:
                cell.locs = locs

                if remove_outside is True:
                    cell.remove_locs_outside_cell()

        except:
            print(traceback.format_exc())
            pass

        return cells

    def add_localisations(self, locs, remove_outside = True, parallel=True,
            silence_tqdm=False, progress_callback=None, max_workers=None):

        if isinstance(locs, pd.DataFrame):
            locs = locs.to_records(index=False)

        if isinstance(locs, np.recarray) == False:
            print("Localisations must be a numpy recarray or pandas dataframe")
            return None

        locs = np.unique(locs)

        jobs = [cell for cell in self.data]

        n_jobs = len(jobs)

        if parallel is True and len(jobs) > 5:
            n_cpus = os.cpu_count()
            if isinstance(max_workers, int):
                n_cpus = min(n_cpus, max_workers)
            executor = ProcessPoolExecutor(max_workers=n_cpus)
            compute_jobs = np.array_split(jobs, n_cpus)
        else:
            n_cpus = 1
            executor = ThreadPoolExecutor(max_workers=n_cpus)
            compute_jobs = [[job] for job in jobs]

        print(f"Adding {len(locs)} localisations to {n_jobs} cells using {n_cpus} CPU(s) cores")

        with Manager() as manager:

            progress_list = manager.list()

            with executor:

                futures = [executor.submit(CellList.add_localisation_task,
                    job, locs, remove_outside, progress_list) for job in compute_jobs]

                self.compute_progress(futures, n_jobs, progress_list, progress_callback,
                    silence_tqdm=silence_tqdm, tqdm_dec="Adding Localisations")

                for future in as_completed(futures):
                    try:
                        results = future.result()

                        if results is not None:
                            if len(results) > 0:
                                for cell in results:
                                    cell_index = cell.cell_index
                                    locs = cell.locs

                                    if type(locs) == np.recarray:
                                        self.data[cell_index] = cell
                                    else:
                                        self.data[cell_index].locs = None

                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()

        locs = self.get_locs()

        if locs is None:
            print(f"Added 0 localisations to CellList")

        if len(locs) > 0:
            n_locs = len(locs)
        else:
            n_locs = 0

        print(f"Added {n_locs} localisations to CellList")

    @staticmethod
    def compute_transforms(jobs, compute_list):
        
        transformed_cells = []

        for job_index, job in enumerate(jobs):

            cell, target_cell, method, shape_measurements = job
            cell = cell_coordinate_transformation(cell=cell, target_cell=target_cell,
                method=method, shape_measurements=shape_measurements)

            if cell is not None:
                cell.transformed = True
                transformed_cells.append(cell)

            compute_list.append(1)

        return transformed_cells
        

    def is_optimised(self, max_error = 5):

        optimised = True

        for cell in self.data:
            if cell.fit_error is None:
                optimised = False
            if cell.fit_error > max_error:
                optimised = False

        return optimised


    def compute_progress(self, futures, n_jobs, progress_list,
            progress_callback=None, silence_tqdm=False, tqdm_dec="Optimising Cells"):

        try:
            last_progress = None
            with tqdm(total=n_jobs, desc=tqdm_dec, disable=silence_tqdm) as pbar:
                while True:
                    progress = ((len(progress_list) + 1) / n_jobs)
                    progress = int(progress * 100)

                    if progress != last_progress:
                        pbar.update(progress - (last_progress or 0))
                        last_progress = progress
                        if progress_callback is not None:
                            progress_callback.emit(progress)

                    time.sleep(0.1)  # Sleep for a short duration to reduce CPU usage

                    if all(future.done() for future in futures):
                        if progress_callback is not None:
                            progress_callback.emit(100)
                        pbar.update(n_jobs - pbar.n)
                        break
        except:
            print(traceback.format_exc())
            return

    def transform_cells(self, target_cell=None, method = "angular",
            shape_measurements=True, progress_callback=None,
            silence_tqdm=False, max_workers=None):

        if target_cell is not None:

            cell_with_locs = [cell for cell in self.data if cell.locs is not None]

            jobs = [list([cell,target_cell,
                          method, shape_measurements]) for cell in cell_with_locs if len(cell.locs) > 0]

            n_jobs = len(jobs)
            n_transformed = 0

            if isinstance(jobs[0][0], Cell) and n_jobs > 1:
                n_cpus = os.cpu_count()
                if isinstance(max_workers, int):
                    n_cpus = min(n_cpus, max_workers)
                executor = ProcessPoolExecutor(max_workers=n_cpus)
                compute_jobs = np.array_split(jobs, n_cpus)
            else:
                n_cpus = 1
                executor = ThreadPoolExecutor(max_workers=n_cpus)
                compute_jobs = [[job] for job in jobs]

            print(f"Transforming localisations within {n_jobs} cells using {n_cpus} CPU(s) cores")

            with Manager() as manager:

                progress_list = manager.list()

                with executor:

                    futures = [executor.submit(CellList.compute_transforms, job, progress_list) for job in compute_jobs]

                    self.compute_progress(futures, n_jobs, progress_list, progress_callback,
                        silence_tqdm=silence_tqdm, tqdm_dec="Transforming Cells")

                    for future in as_completed(futures):

                        try:
                            results = future.result()

                            if results is not None:

                                if len(results) > 0:
                                    for cell in results:
                                        cell_index = cell.cell_index
                                        locs = cell.locs

                                        if type(locs) == np.recarray:
                                            self.data[cell_index] = cell
                                            n_transformed += 1
                                        else:
                                            self.data[cell_index].locs = None

                        except Exception as e:
                            print(f"Error: {e}")
                            traceback.print_exc()

            print(f"Transformed {n_transformed}/{n_jobs} cells")

            for cell in self.data:
                cell.cell_polygon = target_cell.cell_polygon
                cell.cell_midline = target_cell.cell_midline
                cell.cell_radius = target_cell.cell_radius
                cell.remove_locs_outside_cell()

            if len(self.data) > 0:
                self.assign_cell_indices(reindex=True)
                self.assign_cell_names()

        return self

    def get_locs(self, symmetry=False, transformed=None):

        locs = []

        for cell in self.data:
            try:
                cell_locs = cell.locs

                if cell_locs is None:
                    continue
                if len(cell_locs) == 0:
                    continue

                if transformed is not None:
                    if cell.transformed != transformed:
                        continue

                if isinstance(cell_locs, np.recarray):
                    locs.append(cell_locs)

            except:
                continue

        if len(locs) > 0:

            locs = np.hstack(locs).view(np.recarray).copy()
            locs = np.unique(locs)

            if symmetry:
                midline = self.data[0].cell_midline
                centroid_coords = np.array(midline.centroid.coords[0])

                locs = [loc for loc in locs]

                for loc_index in range(len(locs)):
                    loc = locs[loc_index].copy()
                    rloc = reflect_loc_horizontally(loc, centroid_coords)
                    locs.append(rloc)
                for loc_index in range(len(locs)):
                    loc = locs[loc_index].copy()
                    rloc = reflect_loc_vertically(loc, centroid_coords)
                    locs.append(rloc)

                locs = np.hstack(locs).view(np.recarray).copy()

            return locs
        else:
            return None

    def get_cell_lengths(self):

        self.cell_lengths = []

        for cell in self.data:

            try:
                radius = cell.cell_radius
                cell_midline = cell.cell_midline
                midline_length = cell_midline.length
                length = midline_length + (radius * 2)
                pixel_size_nm = cell.pixel_size
                length_um = length * pixel_size_nm / 1000
                cell.cell_length_um = length_um

                self.cell_lengths.append(length_um)
            except:
                print(traceback.format_exc())
                pass

        return self.cell_lengths

    def filter_by_length(self, min_length = 0, max_length = 0):

        if hasattr(self, "cell_lengths") == False:
            self.get_cell_lengths()

        filtered_cells = []

        for i, cell in enumerate(self.data):

            try:
                length_um = cell.cell_length_um

                if length_um >= min_length and length_um <= max_length:
                    filtered_cells.append(cell)
            except:
                pass

        filtered_cells = CellList(filtered_cells)

        return filtered_cells

    def plot_cell_heatmap(self, locs=None, color="red"):

        plot_locs = []

        for cell in self.data:
            locs = cell.locs
            if locs is not None and len(locs) > 0:
                plot_locs.append(locs)

        if len(plot_locs) > 0:
            locs = np.hstack(plot_locs).view(np.recarray).copy()

            # create heatmap
            heatmap, xedges, yedges = np.histogram2d(locs["x"], locs["y"], bins=30, density=False)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.rcParams["axes.grid"] = False
            plt.imshow(heatmap.T, extent=extent, origin='lower')
            plt.show()

    def plot_cell_render(self, oversampling=10, pixel_size=1, blur_method = "One-Pixel-Blur",
            locs=None, color="red"):

        plot_locs = []

        for cell in self.data:
            locs = cell.locs
            if locs is not None and len(locs) > 0:
                plot_locs.append(locs)

        if len(plot_locs) > 0:
            locs = np.hstack(plot_locs).view(np.recarray).copy()

            print(f"Rendering {len(locs)} localisations")

            xmin, xmax = int(np.min(locs["x"])), int(np.max(locs["x"]))
            ymin, ymax = int(np.min(locs["y"])), int(np.max(locs["y"]))

            h = ymax-ymin
            w = xmax-xmin

            viewport = [(float(0), float(0)), (float(h), float(w))]
            image_shape = (1, int(h), int(w))

            if blur_method == "One-Pixel-Blur":
                blur_method = "smooth"
            elif blur_method == "Global Localisation Precision":
                blur_method = "convolve"
            elif blur_method == "Individual Localisation Precision, iso":
                blur_method = "gaussian_iso"
            elif blur_method == "Individual Localisation Precision":
                blur_method = "gaussian"
            else:
                blur_method = None

            # n_rendered_locs, image = render(locs, viewport=viewport, blur_method=blur_method,
            #     min_blur_width=0, oversampling=oversampling, ang=0, )
            #
            # plt.imshow(image)
            # plt.show()

    def add_image(self, image, image_name):

        for cell in self.data:
            cell.add_image(image, image_name)


    def get_custom_cmap(self, colour = "jet"):

        cmap = plt.get_cmap(colour.lower())
        new_cmap = cmap(np.arange(cmap.N))

        new_cmap[0] = [0, 0, 0, 1]

        new_cmap = ListedColormap(new_cmap)

        return new_cmap

    def plot_heatmap(self, symmetry=False, bins=20, cmap="jet",
            draw_outline=True, show=True, save=False, path=None,dpi=500):

        heatmap = None

        try:
            locs = self.get_locs(symmetry=symmetry, transformed=True)

            if locs is None:
                return None

            if len(locs) == 0:
                return None

            polygon = self.data[0].cell_polygon
            polygon_coords = np.array(polygon.exterior.coords)

            n_cells = len(self.data)
            n_locs = len(locs)

            if symmetry:
                n_locs = int(n_locs/4)

            print(f"Generating Cell heatmap with {n_locs} localisations from {n_cells} cells")

            heatmap, xedges, yedges = np.histogram2d(locs["x"], locs["y"], bins=bins)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            cmap = self.get_custom_cmap(colour=cmap)

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap)
            if draw_outline:
                ax.plot(*polygon_coords.T, color='white', linewidth=1)
            ax.axis('off')

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)

            # Set the face color of the colorbar to black and the tick color to white
            cax.set_facecolor('black')
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            buf = BytesIO()
            plt.savefig(buf, format='png',
                bbox_inches='tight', pad_inches=0.1,
                facecolor='black', dpi=dpi)
            buf.seek(0)
            heatmap = plt.imread(buf)

            plt.close(fig)

            if show:
                plt.imshow(heatmap)
                plt.axis('off')
                plt.show()

            if save:
                try:
                    if path is None:
                        print("Save path must be specified")
                        return None
                    plt.imsave(path, heatmap)
                    print(f"Saved heatmap to {path}")
                except:
                    print(traceback.format_exc())
                    return None

            plt.close()

        except:
            print(traceback.format_exc())
            pass

        return heatmap

    def plot_render(self, symmetry=False, oversampling=10, pixel_size=1,
            blur_method="One-Pixel-Blur", min_blur_width=0.2,
            show=True, save=False, path=None, dpi=500,
            cmap="jet", draw_outline=True):

        image = None

        try:

            blur_methods = ["One-Pixel-Blur", "Global Localisation Precision",
                "Individual Localisation Precision, iso", "Individual Localisation Precision"]

            if blur_method not in blur_methods:
                print(f"Blur method must be one of: {blur_methods}")
                return None

            locs = self.get_locs(symmetry=symmetry, transformed=True)

            if locs is None:
                return None

            if len(locs) == 0:
                return None

            polygon = self.data[0].cell_polygon
            polygon_coords = np.array(polygon.exterior.coords)

            locs = pd.DataFrame(locs)

            picasso_columns = ["frame",
                               "y", "x",
                               "photons", "sx", "sy", "bg",
                               "lpx", "lpy",
                               "ellipticity", "net_gradient",
                               "group", "iterations", ]

            column_filter = [col for col in picasso_columns if col in locs.columns]

            locs = locs[column_filter]
            locs = locs.to_records(index=False)

            xmin, xmax = polygon_coords[:, 0].min(), polygon_coords[:, 0].max()
            ymin, ymax = polygon_coords[:, 1].min(), polygon_coords[:, 1].max()

            h,w = int(ymax-ymin)+3, int(xmax-xmin)+3

            viewport = [(float(0), float(0)), (float(h), float(w))]

            if blur_method == "One-Pixel-Blur":
                blur_method = "smooth"
            elif blur_method == "Global Localisation Precision":
                blur_method = "convolve"
            elif blur_method == "Individual Localisation Precision, iso":
                blur_method = "gaussian_iso"
            elif blur_method == "Individual Localisation Precision":
                blur_method = "gaussian"
            else:
                blur_method = None

            print(f"Rendering {len(locs)} localisations")

            n_rendered_locs, image = render(locs,
                viewport=viewport,
                blur_method=blur_method,
                min_blur_width=min_blur_width,
                oversampling=oversampling, ang=0, )

            cmap = self.get_custom_cmap(colour=cmap)

            plt.rcParams["axes.grid"] = False
            fig, ax = plt.subplots()
            ax.imshow(image, cmap=cmap)
            if draw_outline:
                polygon_coords = polygon_coords * oversampling
                ax.plot(*polygon_coords.T, color='white')
            ax.axis('off')

            buf = BytesIO()
            plt.savefig(buf, format='png',
                bbox_inches='tight',pad_inches=0,
                facecolor='black', dpi=dpi)
            buf.seek(0)
            image = plt.imread(buf)
            plt.close(fig)

            if show:
                plt.imshow(image)
                plt.axis('off')
                plt.show()

            if save:
                try:
                    if path is None:
                        print("Save path must be specified")
                        return None
                    plt.imsave(path, image)
                    print(f"Saved render to {path}")
                except:
                    print(traceback.format_exc())
                    return None

            plt.close()

        except:
            print(traceback.format_exc())
            return None

        return image

    def plot_cells(self,xlim=None,ylim=None,locs=True):

        try:

            for cell in copy.deepcopy(self.data):

                polygon = cell.cell_polygon
                polygon_coords = np.array(polygon.exterior.coords)

                plt.plot(*polygon_coords.T, color="black")

                if locs:
                    if cell.locs is None:
                        continue
                    if len(cell.locs) == 0:
                        continue
                    plt.scatter(cell.locs["x"], cell.locs["y"], color="blue")

            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)

            plt.show()


        except:
            print(traceback.format_exc())
            pass





















