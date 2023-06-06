from typing import List

from PIL import Image, ImageDraw
import matplotlib.cm
import numpy as np
import zarr
from matplotlib import pyplot as plt, cm
import cv2
from shapely.geometry import Polygon
import openslide
import torchstain
from torchvision import transforms
from src.zia.annotations.annotation.annotations import AnnotationParser, AnnotationType, \
    Annotation
from zia.annotations.annotation.roi import Roi, PyramidalLevel

# OPENSLIDE_PATH = r'C:\Program Files\OpenSlide\openslide-win64-20230414\bin'
PATH_TO_FILE = "/home/jkuettner/Pictures/wsi_annotations/annotations_species_comparison/mouse_project/objectsjson/MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006.geojson"

import os

# if hasattr(os, 'add_dll_directory'):
#    # Python >= 3.8 on Windows
#    with os.add_dll_directory(OPENSLIDE_PATH):
#        import openslide
# else:
#    import openslide

PATH_TO_PIC = r"/home/jkuettner/qualiperf/P3-MetFun/data/cyp_species_comparison/all/mouse/CYP2E1/MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006.ndpi"

"""
Reduces the list of shapes. It keeps all toplevel shapes, i.e. shapes that do not contain
another shape
"""


def plot_pic(array):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap=matplotlib.colormaps.get_cmap("binary"))
    plt.show()


def plot_rgb(array):
    fig, ax = plt.subplots(1, 1)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    ax.imshow(array)
    plt.show()


def plot_polygons(polygons: List[Polygon], image_like: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    new_image = Image.fromarray(np.zeros_like(image_like), "L")
    draw = ImageDraw.ImageDraw(new_image)
    for poly in polygons:
        draw.polygon(poly.exterior.coords, outline="white", fill="white")

    ax.imshow(new_image, cmap=matplotlib.colormaps.get_cmap("binary"))
    plt.show()


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

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    torch_normalizer.fit(T(cv2image))


