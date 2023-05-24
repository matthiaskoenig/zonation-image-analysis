import matplotlib.cm
import numpy as np
import openslide
import zarr
from matplotlib import pyplot as plt, cm
import cv2
from shapely.geometry import Polygon

PATH_TO_PIC = "/home/jkuettner/Pictures/wsi/J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.ndpi"
PATH_TO_ZARR = "zarr_files/img2.zarr"


def plot_pic(array):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap=matplotlib.colormaps.get_cmap("binary"))
    plt.show()


def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contours[0][:, 0, :]])


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

    blur = cv2.GaussianBlur(cv2image, (3, 3), 5)

    plot_pic(blur)

    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    plot_pic(thresh)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(_)
    # Create a blank image to draw the contours on
    contour_image = np.zeros_like(thresh, dtype=np.uint8)

    # Draw contours on the blank image
    cv2.drawContours(contour_image, contours, -1, (255), thickness=2)
    plot_pic(contour_image)

    # Reduce Contours

    # coords = tuple(map(tuple, contours[0][:, 0, :].toList()))

    ## get contour array to shapely coordinates

    shapes = [Polygon(transform_contour_to_shapely_coords(contour)) for contour in
              contours]

    shapes.sort(key=lambda x: x.area, reverse=True)

    del shapes[0]

    print(len(shapes))







