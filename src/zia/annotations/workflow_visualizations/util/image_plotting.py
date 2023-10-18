from typing import List

import cv2 as cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from shapely import Polygon


def plot_pic(array, title: str = None, cmap: str = "binary_r"):
    fig: plt.Figure
    fig, ax = plt.subplots(1, 1, dpi=300)
    if title is not None:
        fig.suptitle(title)
    show = ax.imshow(array, cmap=matplotlib.colormaps.get_cmap(cmap))
    #bar = plt.colorbar(show)
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
    fig, ax = plt.subplots(1, 1, dpi=300)
    if transform_to_bgr:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    ax.imshow(array)
    plt.show()
