from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import shapely
import skimage
import zarr
from matplotlib import pyplot as plt
from shapely import Polygon, minimum_bounding_radius, minimum_clearance
from skimage.feature import peak_local_max
from skimage import measure

from zia.io.wsi_tifffile import read_ndpi
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.common.slicing import get_tile_slices

import scipy.ndimage as ndi


def detect_droplets_on_tile(tile: np.ndarray) -> List[Polygon]:
    gs = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # blurrs image but preserves edges
    bilateral_filter = cv2.bilateralFilter(src=gs, d=15, sigmaColor=50, sigmaSpace=75)

    # threshold to get the white areas
    _, thresholded = cv2.threshold(bilateral_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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


def circularity(polygon: Polygon) -> float:
    return polygon.length ** 2 / (4 * np.pi * polygon.area)


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


if __name__ == "__main__":
    path = Path(r"D:\image_data\steatosis\RoiExtraction\FLR-180\0\J-15-0789_FLR-180_Lewis_HE_Run 06_LLL02_MAA_003.ome.tiff")
    array = read_ndpi(path)[0]

    print(array.shape)

    sub_array = array[8750: 8750 + 1000, 12500: 12500 + 2048]
    plot_pic(sub_array)

    gs = cv2.cvtColor(sub_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    plot_pic(gs)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gs = clahe.apply(gs)
    # plot_pic(gs)

    # blurrs image but preserves edges
    bilateral_filter = cv2.bilateralFilter(src=gs, d=15, sigmaColor=50, sigmaSpace=75)
    # plot_pic(bilateral_filter)

    # threshold to get the white areas
    _, thresholded = cv2.threshold(bilateral_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot_pic(thresholded)

    ## reduces noise -> to be adapted to not remove any meaningful data
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    plot_pic(opening, "cleaned up mask")

    ## get the contours from the opened binary mask
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = [Polygon(np.squeeze(cnt)) for cnt in contours]

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

    # integrate_droplets(polygons)

    circular, non_circular = filter_solidity(polygons)

    circular_cnts = [np.array(poly.exterior.coords, dtype=np.int32) for poly in circular]
    non_circular_cnts = [np.array(poly.exterior.coords, dtype=np.int32) for poly in non_circular]

    cv2.drawContours(sub_array, circular_cnts, -1, (0, 255, 0), 2)
    cv2.drawContours(sub_array, non_circular_cnts, -1, (255, 0, 0), 2)

    plot_pic(sub_array)
