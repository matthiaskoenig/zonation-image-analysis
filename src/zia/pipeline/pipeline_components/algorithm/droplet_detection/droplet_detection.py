from pathlib import Path
from typing import List

import cv2
import numpy as np
import skimage
import zarr
from matplotlib import pyplot as plt
from shapely import Polygon
from skimage.feature import peak_local_max
from skimage import measure

from zia.io.wsi_tifffile import read_ndpi
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.common.slicing import get_tile_slices

import scipy.ndimage as ndi


def detect_droplets_on_tile(tile: np.ndarray) -> List[Polygon]:
    gs = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    filtered = None


def detect_droplets(array: zarr.Array):
    slices = get_tile_slices(array.shape[:2], (2 ** 12, 2 ** 12), pad=200)

    for cs, rs in slices:
        tile = array[cs, rs]
        if np.all(tile == 0):
            continue


if __name__ == "__main__":
    path = Path(r"D:\image_data\steatosis\RoiExtraction\FLR-180\0\J-15-0789_FLR-180_Lewis_HE_Run 06_LLL02_MAA_003.ome.tiff")
    array = read_ndpi(path)[0]

    sub_array = array[17500: 17500 + 2048, 25000: 25000 + 2048]
    plot_pic(sub_array)

    gs = cv2.cvtColor(sub_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    plot_pic(gs)

    # blurrs image but preserves edges
    bilateral_filter = cv2.bilateralFilter(src=gs, d=15, sigmaColor=50, sigmaSpace=75)
    plot_pic(bilateral_filter)

    # threshold to get the white areas
    _, thresholded = cv2.threshold(bilateral_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plot_pic(thresholded)

    ## reduces noise -> to be adapted to not remove any meaningful data
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

    plot_pic(opening, "cleaned up mask")

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    plot_pic(dist_transform, "distance transform")

    # finding the local maxima of the distance transform and create markers for water shedding
    plm = peak_local_max(dist_transform, min_distance=10)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(plm.T)] = True
    plot_pic(mask, "peak local max")
    markers, _ = ndi.label(mask)

    segmented = skimage.segmentation.watershed(255 - dist_transform, markers, mask=opening)

    segmented = segmented - 1

    contours, _ = cv2.findContours(segmented, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(sub_array, contours, -1, (0, 255, 0), 2)

    plot_pic(sub_array)

    polygons = [Polygon(np.squeeze(cnt)) for cnt in contours]
