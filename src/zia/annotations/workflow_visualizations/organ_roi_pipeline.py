import os
from typing import List

import cv2
import numpy as np
from shapely.geometry import Polygon

from src.zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from src.zia.annotations.annotation.roi import Roi, PyramidalLevel
from zia.annotations import OPENSLIDE_PATH
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic, \
    plot_polygons

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

PATH_TO_FILE = "/home/jkuettner/Pictures/wsi_annotations/annotations_species_comparison/mouse_project/objectsjson/MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006.geojson"


PATH_TO_PIC = r"/home/jkuettner/qualiperf/P3-MetFun/data/cyp_species_comparison/all/mouse/CYP2E1/MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006.ndpi"

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
        kept_shapes.append(not_in_bigger_shape[0])
    elif len(not_in_bigger_shape) == 0:
        return
    else:
        reduce_shapes(kept_shapes, not_in_bigger_shape)


def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contour[:, 0, :]])


def filter_shapes(contours):
    return [contour for contour in contours if contour.shape[0] >= 4]


if __name__ == "__main__":
    image = openslide.OpenSlide(PATH_TO_PIC)

    w, h = image.dimensions

    # print(image.level_count)
    # print(image.level_downsamples)

    level = 7
    factor = image.level_downsamples[level]

    region = image.read_region(location=(0, 0), level=7,
                               size=(int(w / factor), int(h / factor)))

    cv2image = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)

    # pad the image to handle edge cutting tissue regions
    padding = 10

    padded_copy = cv2.copyMakeBorder(cv2image, padding, padding, padding, padding,
                                     cv2.BORDER_CONSTANT, value=255)

    plot_pic(padded_copy)

    blur = cv2.GaussianBlur(padded_copy, (5, 5), 5, borderType=cv2.BORDER_REPLICATE)

    plot_pic(blur)

    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    plot_pic(thresh)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(-padding, -padding))
    # Create a blank image to draw the contours on
    contour_image = np.zeros_like(cv2image, dtype=np.uint8)

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

    # load
    annotations = AnnotationParser.parse_geojson(PATH_TO_FILE)
    liver_annotation_shapes = AnnotationParser.get_annotation_by_type(annotations,
                                                                      annotation_type=AnnotationType.LIVER)

    # find the contour the organ shape that contains the annotation geometry

    liver_shapes = [shape for shape in kept for anno in liver_annotation_shapes
                    if shape.contains(anno.get_resized_geometry(128))]

    liver_rois = [Roi(liver_shape, PyramidalLevel.SEVEN, AnnotationType.LIVER) for
                  liver_shape in liver_shapes]

    plot_polygons(
        [liver_roi.get_polygon_for_level(PyramidalLevel.SEVEN) for liver_roi in
         liver_rois], contour_image)

    ##
    # liver_roi.write_to_geojson("resources/geojsons/result.geojson")
