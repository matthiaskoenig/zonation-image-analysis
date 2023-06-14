from typing import List

from PIL import Image, ImageDraw
import matplotlib.cm
import numpy as np
import zarr
from matplotlib import pyplot as plt, cm
import cv2
from shapely.geometry import Polygon

from src.zia.annotations.normalization.marcenko import normalizeStaining

import os

from zia.annotations import OPENSLIDE_PATH
from zia.annotations.workflow_visualizations.util.image_plotting import plot_rgb, \
    plot_pic

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

PATH_TO_FILE = "/home/jkuettner/Pictures/wsi_annotations/annotations_species_comparison/mouse_project/objectsjson/MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006.geojson"



PATH_TO_PIC = r"D:\data\cyp_species_comparison\all\mouse\CYP2E1/MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006.ndpi"

"""
Reduces the list of shapes. It keeps all toplevel shapes, i.e. shapes that do not contain
another shape
"""

def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contour[:, 0, :]])


def filter_shapes(contours):
    return [contour for contour in contours if contour.shape[0] >= 4]


if __name__ == "__main__":
    image = openslide.OpenSlide(PATH_TO_PIC)

    w, h = image.dimensions

    # print(image.level_count)
    # print(image.level_downsamples)

    level = 3
    factor = image.level_downsamples[level]

    region = image.read_region(location=(100 * 128, 400 * 128), level=level,
                               size=(512, 512))

    cv2image = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
    plot_rgb(cv2image)

    RC1, RC1N, RC2, RC2N = normalizeStaining(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB), Io=240, alpha=1, beta=0.15)

    plot_pic(RC2)
    plot_pic(RC1)
    #plot_rgb(INorm, transform_to_bgr=False)
