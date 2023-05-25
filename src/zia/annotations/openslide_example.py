from typing import List

from PIL import Image, ImageDraw
import matplotlib.cm
import numpy as np
import zarr
from matplotlib import pyplot as plt, cm
import cv2
from shapely.geometry import Polygon

OPENSLIDE_PATH = r'C:\Program Files\OpenSlide\openslide-win64-20230414\bin'

import os

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

PATH_TO_PIC = r"C:\Users\jonas\Development\images\J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.ndpi"
PATH_TO_ZARR = "zarr_files/img2.zarr"

"""
Reduces the list of shapes. It keeps all toplevel shapes, i.e. shapes that do not contain
another shape
"""


def reduce_shapes(kept_shapes: List[Polygon], remaining_shapes: List[Polygon]) -> None:
    not_in_bigger_shape = []
    big_shape = remaining_shapes.pop(0)
    kept_shapes.append(big_shape)
    for smaller_shape in remaining_shapes:
        if not smaller_shape.within(big_shape):
            not_in_bigger_shape.append(smaller_shape)

    if len(not_in_bigger_shape) == 1:
        kept.append(not_in_bigger_shape[0])
    elif len(not_in_bigger_shape) == 0:
        return
    else:
        reduce_shapes(kept_shapes, not_in_bigger_shape)


def plot_pic(array):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap=matplotlib.colormaps.get_cmap("binary"))
    plt.show()


def plot_polygons(polygons: List[Polygon], image_like: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    new_image = Image.fromarray(np.zeros_like(image_like), "L")
    draw = ImageDraw.ImageDraw(new_image)
    for poly in polygons:
        print(poly)
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

    print(image.level_count)
    print(image.level_downsamples)

    level = 7
    factor = image.level_downsamples[level]
    print((w / factor, h / factor))
    region = image.read_region(location=(0, 0), level=7,
                               size=(int(w / factor), int(h / factor)))

    cv2image = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)

    plot_pic(cv2image)

    blur = cv2.GaussianBlur(cv2image, (5, 5), 5)

    plot_pic(blur)

    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    plot_pic(thresh)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Create a blank image to draw the contours on
    contour_image = np.zeros_like(thresh, dtype=np.uint8)

    # Draw contours on the blank image
    cv2.drawContours(contour_image, contours, -1, (255), thickness=2)
    plot_pic(contour_image)

    # get contour array to shapely coordinates
    contours = filter_shapes(contours)
    shapes = [Polygon(transform_contour_to_shapely_coords(contour)) for contour in
              contours]

    # drop shapes which are within a bigger shape
    shapes.sort(key=lambda x: x.area, reverse=True)
    kept = []
    remaining = shapes[1:]  # removing image bounding box
    reduce_shapes(kept, remaining)
    plot_polygons(kept, contour_image)

    # check if organ annotation is inside one of them
