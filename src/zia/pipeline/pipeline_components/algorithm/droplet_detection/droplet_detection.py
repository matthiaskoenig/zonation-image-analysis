from pathlib import Path
from typing import List, Tuple, Callable

import cv2
import numpy as np
import shapely
import skimage
import zarr
from matplotlib import pyplot as plt
from shapely import Polygon, minimum_bounding_radius, minimum_clearance, minimum_rotated_rectangle
from skimage.feature import peak_local_max
from skimage import filters
from skimage.filters.thresholding import apply_hysteresis_threshold

from zia.io.wsi_tifffile import read_ndpi
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.common.slicing import get_tile_slices

import scipy.ndimage as ndi


def get_foreground_mask(array: np.ndarray) -> np.ndarray:
    gs = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # blurrs image but preserves edges
    bilateral_filter = cv2.bilateralFilter(src=gs, d=5, sigmaColor=50, sigmaSpace=75)

    # bilateral_filter = cv2.GaussianBlur(gs, (11, 11), sigmaX=5, sigmaY=5)
    plot_pic(bilateral_filter)

    # threshold to get the white areas
    thresholded = apply_hysteresis_threshold(bilateral_filter, 170, 200).astype(np.uint8)*255

    #print(thresholded)
    #print(thresholded)
    #_, thresholded = cv2.threshold(bilateral_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #kernel = np.ones((3, 3), np.uint8)
    #opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    #_, thresholded = cv2.threshold(bilateral_filter, 200, 255, cv2.THRESH_BINARY)
    return thresholded

def detect_droplets_on_tile(tile: np.ndarray) -> List[Polygon]:

    thresholded = get_foreground_mask(tile)

    ## reduces noise -> to be adapted to not remove any meaningful data
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # finding the local maxima of the distance transform and create markers for water shedding
    plm = peak_local_max(dist_transform, min_distance=10)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(plm.T)] = True
    markers, _ = ndi.label(mask)

    segmented = skimage.segmentation.watershed(255 - dist_transform, markers, mask=opening)
    segmented = segmented - 1
    contours, _ = cv2.findContours(segmented, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    polys = [Polygon(np.squeeze(cnt)) for cnt in contours]

    circular, non_circular = filter_solidity(polys)

    return circular


def solidity(polygon: Polygon) -> float:
    return polygon.area / polygon.convex_hull.area


def circularity(polygon: Polygon) -> float:
    return (4 * np.pi * polygon.area) / polygon.length ** 2


def roundness(polygon: Polygon) -> float:
    return (4 * np.pi * polygon.area) / (polygon.convex_hull.length) ** 2


def convexity(polygon: Polygon) -> float:
    return polygon.convex_hull.length / polygon.exterior.length


def sphericity(polygon: Polygon) -> float:
    pcoords = np.vstack(polygon.exterior.xy).T
    centroid = [polygon.centroid.x, polygon.centroid.y]

    r_inner = np.min(np.linalg.norm(pcoords - centroid, axis=1))
    r_outer = minimum_bounding_radius(polygon)
    return r_inner / r_outer


def elongation(polygon: Polygon) -> float:
    bounding_rect = minimum_rotated_rectangle(polygon)

    min_x, min_y, max_x, max_y = bounding_rect.bounds

    width = max_x - min_x
    height = max_y - min_y

    if height > width:
        return width / height

    return height / width


def compactness(polygon: Polygon) -> float:
    return 4 * np.pi * polygon.area / polygon.length ** 2


def feature_vector(p: Polygon, feature_funs: List[Callable[[Polygon], float]]) -> List[float]:
    return [fun(p) for fun in feature_funs]


def clean_droplets(polygons: List[Polygon]) -> Tuple[List[Polygon], List[Polygon]]:
    polygons = [shapely.simplify(p, 3) for p in polygons]
    circularity_filtered = list(filter(lambda p: circularity(p) < 1.5, polygons))

    return circularity_filtered, [p for p in polygons if p not in circularity_filtered]


def filter_circularity(polygons: List[Polygon]) -> List[Polygon]:
    return list(filter(lambda p: circularity(p) < 3, polygons))


def detect_droplets(array: zarr.Array):
    slices = get_tile_slices(array.shape[:2], (2 ** 12, 2 ** 12), pad=200)

    for cs, rs in slices:
        tile = array[cs, rs]
        if np.all(tile == 0):
            continue


def filter_solidity(polygons: List[Polygon]) -> Tuple[List[Polygon], List[Polygon]]:
    _solid = list(filter(lambda p: p.area / p.convex_hull.area > 0.9, polygons))
    _non_solid = list(filter(lambda p: p not in _solid, polygons))
    return _solid, _non_solid


def filter_size(polygons: List[Polygon], min_diameter=6, pixel_size=0.22) -> List[Polygon]:
    return list(filter(lambda p: p.area >= np.pi * (min_diameter / pixel_size), polygons))


if __name__ == "__main__":
    path = Path("/media/jkuettner/Extreme Pro/image_data/steatosis/RoiExtraction/FLR-180/0/J-15-0789_FLR-180_Lewis_HE_Run 06_LLL02_MAA_003.ome.tiff")
    array = read_ndpi(path)[0]

    print(array.shape)

    sub_array = array[8750: 8750 + 2048, 12500: 12500 + 2048]
    plot_pic(sub_array)

    gs = cv2.cvtColor(sub_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    plot_pic(gs)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gs = clahe.apply(gs)

    plt.show()    # plot_pic(gs)

    # blurrs image but preserves edges
    bilateral_filter = cv2.bilateralFilter(src=gs, d=15, sigmaColor=50, sigmaSpace=75)

    # bilateral_filter = cv2.GaussianBlur(gs, (11, 11), sigmaX=5, sigmaY=5)
    plot_pic(bilateral_filter)




    # threshold to get the white areas
    _, thresholded = cv2.threshold(bilateral_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot_pic(thresholded)

    ## reduces noise -> to be adapted to not remove any meaningful data
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    plot_pic(opening, "cleaned up mask")

    ## get the contours from the opened binary mask
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # plot_pic(sub_array)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # plot_pic(dist_transform, "distance transform")

    # finding the local maxima of the distance transform and create markers for water shedding
    plm = peak_local_max(dist_transform, min_distance=10)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(plm.T)] = True
    # plot_pic(mask, "peak local max")
    markers, _ = ndi.label(mask)

    segmented = skimage.segmentation.watershed(255 - dist_transform, markers, mask=opening)

    segmented = segmented - 1

    contours, _ = cv2.findContours(segmented, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = [Polygon(np.squeeze(cnt)) for cnt in contours]
    polygons = filter_size(polygons)

    feature_funs = [circularity, roundness, convexity, sphericity, elongation, compactness, solidity]

    feature_vectors = np.vstack([feature_vector(p, feature_funs) for p in polygons])

    print(feature_vectors)
    # integrate_droplets(polygons)

    circular, non_circular = filter_solidity(polygons)

    circular_cnts = [np.array(poly.exterior.coords, dtype=np.int32) for poly in circular]
    non_circular_cnts = [np.array(poly.exterior.coords, dtype=np.int32) for poly in non_circular]

    cv2.drawContours(sub_array, circular_cnts, -1, (0, 255, 0), 2)
    cv2.drawContours(sub_array, non_circular_cnts, -1, (255, 0, 0), 2)

    plot_pic(sub_array)
