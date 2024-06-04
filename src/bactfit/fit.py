import tifffile 
import numpy as np
import cv2
import os
import traceback
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis, binary_opening, disk
from scipy.interpolate import CubicSpline, BSpline
from skimage.draw import polygon
from shapely.geometry import LineString, Point, LinearRing, Polygon
from shapely.affinity import rotate
import matplotlib.path as mpltPath
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
from scipy.interpolate import interp1d
import math
from scipy.interpolate import splrep, splev
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import shapely
from scipy.spatial.distance import directed_hausdorff
import shapely
from scipy.spatial import distance
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Event
from functools import partial


class bactfit:
    
    def __init__(self, verbose = True):
        
        self.contours = []
        self.cell_masks = []
        self.verbose = verbose


    def fit_cells(self, masks = [], fit = True, parallel = False):

        if type(masks) != list:
            masks = [masks]

        self.populate_cell_dataset(masks, edge_cells=False)

        self.run(fit = fit,
            parallel = parallel)


    def check_edge_cell(self, cnt, mask, buffer=5):

        edge = False

        try:

            cell_mask_bbox = cv2.boundingRect(cnt)
            [x, y, w, h] = cell_mask_bbox
            [x1, y1, x2, y2] = [x, y, x + w, y + h]
            bx1, by1, bx2, by2 = [x1 - buffer, y1 - buffer, x2 + buffer, y2 + buffer]

            if bx1 < 0:
                edge = True
            if by1 < 0:
                edge = True
            if bx2 > mask.shape[1]:
                edge = True
            if by2 > mask.shape[0]:
                edge = True

        except:
            print(traceback.format_exc())

        return edge, [bx1, by1, bx2, by2]
    
    def populate_cell_dataset(self, masks, edge_cells=False):
        
        self.cell_dataset = []

        for mask_index, mask in enumerate(masks):

            mask_ids = np.unique(mask)

            for mask_id in mask_ids:

                try:

                    if mask_id != 0:

                        cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                        cell_mask[mask == mask_id] = 1

                        cnt, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cnt = cnt[0]

                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        x, y, w, h = cv2.boundingRect(cnt)

                        edge, crop_coords = self.check_edge_cell(cnt, mask)

                        x1,y1,x2,y2 = crop_coords
                        cell_mask_crop = cell_mask[y1:y2,x1:x2]

                        cnt, _ = cv2.findContours(cell_mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cnt = cnt[0]

                        if h > w:
                            vertical = True
                        else:
                            vertical = False

                        dat = {"mask_index":mask_index,
                               "mask_id":mask_id,
                               "edge":edge,
                               "bbox":[x1,y1,x2,y2],
                               "cell_centre":[cx,cy],
                               "cnt":cnt,
                               "cell_mask": cell_mask_crop,
                               "vertical":vertical,
                               }

                        if edge_cells == False:
                            if edge == False:
                                self.cell_dataset.append(dat)

                        else:
                            self.cell_dataset.append(dat)

                except:
                    pass
        
        if self.verbose:
            print(f"Imported {len(self.cell_dataset)} cells")
            
        return self.cell_dataset

    @staticmethod
    def get_medial_axis_coords(cell_mask, cell_contour = None,
            refine = True, iterations = 3):
        
        coords_list = []
        
        for i in range(iterations):
            skeleton = medial_axis(cell_mask, return_distance=False)
            coords = np.flip(np.transpose(np.nonzero(skeleton)),axis=1)
            coords_list.append(coords)
            
        coords = np.vstack(coords_list)
        coords = np.unique(coords, axis=0)
         
        polygon = Polygon(cell_contour.coords)

        centroid = polygon.centroid
        cell_radius = cell_contour.distance(centroid)

        if refine:
            coords = [p for p in coords if cell_contour.distance(Point(p)) > cell_radius*0.8]
            coords = np.array(coords)

        return coords, cell_radius

    @staticmethod
    def resize_line(line, length):
        
        distances = np.linspace(0, line.length, length)
        line = LineString([line.interpolate(distance) for distance in distances])
    
        return line

    @staticmethod
    def moving_average(line, padding=5, iterations=1):
        
        x, y = line[:, 0], line[:, 1]

        x = np.concatenate((x[-padding:], x, x[:padding]))
        y = np.concatenate((y[-padding:], y, y[:padding]))

        for i in range(iterations):
            y = np.convolve(y, np.ones(padding), 'same') / padding
            x = np.convolve(x, np.ones(padding), 'same') / padding

            x = np.array(x)
            y = np.array(y)

        x = x[padding:-padding]
        y = y[padding:-padding]

        line = np.stack([x, y]).T

        return line

    @staticmethod
    def get_cell_contour(cnt, smooth = True):
        
        cell_contour_coords = cnt.reshape(-1,2)
        
        if smooth:
            cell_contour_coords = bactfit.moving_average(cell_contour_coords)
        
        cell_contour = LinearRing(cell_contour_coords)
        cell_contour = cell_contour.buffer(0.5)
        cell_contour = LinearRing(cell_contour.exterior.coords)
        cell_contour = bactfit.resize_line(cell_contour, 100)
        
        return cell_contour

    @staticmethod
    def bactfit_result(params, cell_contour, poly_params,
            fit_mode = "directed_hausdorff", containment_penalty = False):
        
        x_min = params[0]
        x_max = params[1]
        cell_width = params[2]
        x_offset = params[3]
        y_offset = params[4]

        p = np.poly1d(poly_params)
        x_fitted = np.linspace(x_min, x_max, num=20)
        y_fitted = p(x_fitted)
        
        x_fitted += x_offset
        y_fitted += y_offset
    
        midline_coords = np.column_stack((x_fitted,y_fitted))
    
        midline = LineString(midline_coords)
    
        cell_model = midline.buffer(cell_width)

        distance = bactfit.compute_bacfit_distance(cell_model, cell_contour,
            fit_mode, containment_penalty)

        x = midline_coords[:,0]
        y = midline_coords[:,1]
        bisector_coords = bactfit.get_poly_coords(x, y, 
                                            poly_params, 
                                            n_points=100, 
                                            margin=cell_width*2)
        
        bisector = LineString(bisector_coords)
        
        fit_data = {"cell_model":cell_model,
                    "cell_midline":midline,
                    "cell_width":cell_width,
                    "bisector":bisector,
                    "error":distance,
                    }
 
        return fit_data

    @staticmethod
    def compute_bacfit_distance(midline_buffer, cell_contour,
            fit_mode = "directed_hausdorff", containment_penalty = False):

        try:

            if fit_mode == "hausdorff":
                # Calculate the Hausdorff distance between the buffered spline and the target contour
                distance = midline_buffer.hausdorff_distance(cell_contour)
            elif fit_mode == "directed_hausdorff":
                # Calculate directed Hausdorff distance in both directions
                buffer_points = np.array(midline_buffer.exterior.coords)
                contour_points = np.array(cell_contour.coords)
                dist1 = directed_hausdorff(buffer_points, contour_points)[0]
                dist2 = directed_hausdorff(contour_points, buffer_points)[0]
                distance = dist1 + dist2

            if containment_penalty:
                if cell_contour.contains(midline_buffer) == False:
                    distance = distance * 1.5

        except:
            distance = np.inf

        return distance


    @staticmethod
    def bactfit_function(params, cell_contour, poly_params,
            fit_mode ="directed_hausdorff", containment_penalty=False):
        
        """
        Objective function to minimize: the Hausdorff distance between the buffered spline and the target contour.
        """
        
        try:
            
            params = list(params)
            
            x_min = params[0]
            x_max = params[1]
            cell_width = params[2]
            x_offset = params[3]
            y_offset = params[4]

            p = np.poly1d(poly_params)
            x_fitted = np.linspace(x_min, x_max, num=10)
            y_fitted = p(x_fitted)
            
            x_fitted += x_offset
            y_fitted += y_offset

            midline_coords = np.column_stack((x_fitted,y_fitted))
        
            midline = LineString(midline_coords)

            midline_buffer = midline.buffer(cell_width)

            distance = bactfit.compute_bacfit_distance(midline_buffer, cell_contour,
                fit_mode, containment_penalty)

        except:
            print(traceback.format_exc())
            distance = np.inf

        return distance

    @staticmethod
    def get_poly_coords(x, y, coefficients, 
                        margin = 0, n_points=10):
        
        x1 = np.min(x) - margin
        x2 = np.max(x) + margin
        
        p = np.poly1d(coefficients)
        x_fitted = np.linspace(x1, x2, num=n_points)
        y_fitted = p(x_fitted)

        return np.column_stack((x_fitted,y_fitted))

    @staticmethod
    def fit_poly(coords, degree = 2, vertical = False,
                 constrained = True, 
                 constraining_points = [], 
                 minimise_curvature = False, maxiter = 50):
        
        def polynomial_fit(params, x):
            # Reverse the parameters to match np.polyfit order
            params = params[::-1]
            return sum(p * x**i for i, p in enumerate(params))

        def objective_function(params, x, y, minimise_curvature=True):
            
            fit_error = np.sum((polynomial_fit(params, x) - y)**2)
            
            if minimise_curvature:
                curvature_penalty = np.sum(np.diff(params, n=2)**2)
                
                fit_error = fit_error + curvature_penalty
            
            return fit_error

        def constraint_function(params, x_val, y_val):
            return polynomial_fit(params, x_val) - y_val

        def get_coords(x, y, coefficients, margin = 0, n_points=10):
            
            x1 = np.min(x) - margin
            x2 = np.max(x) + margin
            
            p = np.poly1d(coefficients)
            x_fitted = np.linspace(x1, x2, num=n_points)
            y_fitted = p(x_fitted)

            return np.column_stack((x_fitted,y_fitted))
            
        x = coords[:,0]
        y = coords[:,1]
        constraints = []
        
        param_list = []
        error_list = []
        success_list = []
        
        if constrained and len(constraining_points) > 0:
            
            for point in constraining_points:
                if len(point) == 2:
                    constraints.append({'type': 'eq', 
                                        'fun': constraint_function, 
                                        'args': point})
          
        if type(degree) != list:  
            degree = [degree]
            

        for deg in degree:
            
            params = np.polyfit(x, y, deg)
            
            result = minimize(objective_function, 
                              params, 
                              args=(x, y, minimise_curvature),
                              constraints=constraints, 
                              tol=1e-6, 
                              options={'maxiter': maxiter})
            
            param_list.append(result.x)
            error_list.append(result.fun)
            success_list.append(result.success)
                
        min_error_index = error_list.index(min(error_list))
        
        best_params = param_list[min_error_index]

        fitted_poly = get_coords(x, y, best_params)
            
        return fitted_poly, list(best_params)

    @staticmethod
    def register_line(line, cell_centre, vertical):
        
        x_center = np.mean(line[:, 0])
        y_center = np.mean(line[:, 1])
        
        line[:,0] -= x_center
        line[:,1] -= y_center
        
        if vertical:
            
            x = line[:,0]
            y = line[:,1]
            
            angle = 90
            angle = math.radians(angle)
            
            # Rotation matrix multiplication to get rotated x & y
            xr = (x * math.cos(angle)) - (y * math.sin(angle))
            yr = (x * math.sin(angle)) + (y * math.cos(angle))
            
            line[:, 0] = xr
            line[:, 1] = yr
            
        line[:, 0] += cell_centre[0]
        line[:, 1] += cell_centre[1]
        
        return line

    @staticmethod
    def register_fit_data(fit_data, cell_centre, vertical):
        
        cell_model = fit_data["cell_model"]
        bisector = fit_data["bisector"]
        
        cell_model = np.array(cell_model.exterior.coords)
        bisector = np.array(bisector.coords)
        
        cell_model = bactfit.register_line(cell_model, cell_centre, vertical)
        bisector = bactfit.register_line(bisector, cell_centre, vertical)
        
        cell_model = Polygon(cell_model)
        bisector = LineString(bisector)
        
        fit_data["cell_model"] = cell_model
        fit_data["bisectir"] = bisector

        return fit_data

    @staticmethod
    def fit_cell(cell_data, progress_list = [], fit = True, fit_mode = "directed_hausdorff"):

        fit_data = None

        try:

            cell_mask = cell_data["cell_mask"]
            cell_centre = cell_data["cell_centre"]
            vertical = cell_data["vertical"]

            if vertical:
                cell_mask = cv2.rotate(cell_mask, cv2.ROTATE_90_CLOCKWISE)

            cnt, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = cnt[0]

            cell_contour = bactfit.get_cell_contour(cnt)

            medial_axis_coords, radius = bactfit.get_medial_axis_coords(cell_mask,
                cell_contour, iterations=1)

            medial_axis_fit, poly_params = bactfit.fit_poly(medial_axis_coords,
                degree=[1, 2], maxiter=50)

            x_min = np.min(medial_axis_fit[:, 0])
            x_max = np.max(medial_axis_fit[:, 0])
            x_offset = 0
            y_offset = 0
            params = [x_min, x_max, radius, x_offset, y_offset]

            if fit:
                result = minimize(bactfit.bactfit_function,
                    params,
                    args=(cell_contour, poly_params, fit_mode),
                    tol=1e-12,
                    options={'maxiter': 100})

                params = result.x

            fit_data = bactfit.bactfit_result(params, cell_contour, poly_params)
            fit_data = bactfit.register_fit_data(fit_data, cell_centre, vertical)

            fit_data["mask_id"] = cell_data["mask_id"]
            fit_data["mask_index"] = cell_data["mask_index"]

        except:
            pass

        progress_list.append(1)

        return fit_data

    def run(self, fit = True, parallel = False, max_workers = None,
            progress_callback = None, fit_mode = "directed_hausdorff"):

        fit_results = []

        if len(self.cell_dataset) > 0:

            self.cell_dataset = self.cell_dataset[:50]

            with Manager() as manager:

                progress_list = manager.list()

                num_cells = len(self.cell_dataset)
                fit_func = partial(self.fit_cell,
                    progress_list=progress_list,
                    fit=fit, fit_mode=fit_mode)

                if parallel:

                    if max_workers == None:
                        max_workers = os.cpu_count()

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:

                        futures = [executor.submit(fit_func, cell_data) for cell_data in self.cell_dataset]

                        while any(not future.done() for future in futures):
                            progress = (sum(progress_list) / num_cells)
                            progress = progress * 100

                            if progress_callback is not None:
                                progress_callback.emit(progress)

                        fit_results = [future.result() for future in futures]

                else:

                    for cell_data in tqdm(self.cell_dataset):

                        fit_result = self.fit_cell(cell_data, progress_list, fit)

                        fit_results.append(fit_result)

                        progress = (sum(progress_list) / num_cells)
                        progress = progress * 100

                        if progress_callback is not None:
                            progress_callback.emit(progress)

        return fit_results









                        