from typing import List

import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw
import numpy as np
from shapely import Polygon
import cv2 as cv2


def plot_pic(array):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap=matplotlib.colormaps.get_cmap("binary"))
    plt.show()


def plot_polygons(polygons: List[Polygon], image_like: np.ndarray):
    fig, ax = plt.subplots(1, 1)
    new_image = Image.fromarray(np.zeros_like(image_like), "L")
    draw = ImageDraw.ImageDraw(new_image)
    for poly in polygons:
        draw.polygon(poly.exterior.coords, outline="white", fill="white")

    ax.imshow(new_image, cmap=matplotlib.colormaps.get_cmap("binary"))
    plt.show()

def plot_rgb(array, transform_to_bgr=True):
    fig, ax = plt.subplots(1, 1)
    if transform_to_bgr:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    ax.imshow(array)
    plt.show()
