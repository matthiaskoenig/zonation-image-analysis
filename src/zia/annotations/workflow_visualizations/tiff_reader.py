import os.path
from typing import List, Tuple, Optional

import cv2
import dask.array
import numpy as np
import shapely
import tifffile
import zarr.convenience
from PIL.ImageDraw import ImageDraw
from tifffile.tifffile import ZarrTiffStore
from dask.array import from_zarr

from zia import DATA_PATH, RESULTS_PATH
from zia.annotations import OPENSLIDE_PATH
from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.annotation.roi import Roi, PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic, \
    plot_rgb
from zarr.hierarchy import Group

from tifffile import imread

from zia.annotations.zarr_image.zarr_image import ZarrImage, LeveledRoi

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

IMAGE_NAME = "J-12-00348_NOR-021_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006"

ROI_ANNOTATION_PATH = RESULTS_PATH / "annotations_liver_roi" / "rat"
IMAGE_PATH = DATA_PATH / "cyp_species_comparison" / "all" / "rat" / "CYP2E1"
ZARR_PATH = DATA_PATH / "zarr_files"


def get_annotation_path(image_name: str) -> Optional[str]:
    json_file = image_name + ".geojson"
    base = DATA_PATH / "annotations_species_comparison"
    for subfolder in os.listdir(base):
        image_path = os.path.join(base, subfolder, "objectsjson", json_file)
        if os.path.isfile(image_path):
            return image_path


def get_bounds(level: PyramidalLevel, rois: List[Roi]) -> List[Tuple[slice, slice]]:
    return [get_bound(level, roi) for roi in rois]


def get_bound(level: PyramidalLevel, roi: Roi) -> Tuple[slice, slice]:
    poly = roi.get_polygon_for_level(level)
    b = poly.bounds

    xs = slice(int(b[0]), int(b[2]))
    ys = slice(int(b[1]), int(b[3]))
    return xs, ys


def create_polygon_from_slice_bounds():
    pass


if __name__ == "__main__":

    rois = Roi.load_from_file(
        os.path.join(ROI_ANNOTATION_PATH, f"{IMAGE_NAME}.geojson"))

    zarr_image = ZarrImage(IMAGE_NAME, rois)

    ## reading image with tiffile as zarr store

    for roi_no, (leveled_roi, roi) in enumerate(
        zip(zarr_image.rois, zarr_image._roi_annos)):
        (arr, (x_min, y_min, x_max, y_max)) = leveled_roi.get_by_level(
            PyramidalLevel.ZERO)
        bound_poly = shapely.box(x_min, y_min, x_max, y_max)

        annotations = AnnotationParser.get_annotation_by_types(
            AnnotationParser.parse_geojson(get_annotation_path(IMAGE_NAME)),
            AnnotationType.get_artifacts())

        roi_artifacts = []
        print(len(annotations))

        for anno in annotations:
            if anno.geometry.within(bound_poly):
                roi_artifacts.append(anno)
            elif anno.geometry.intersects(bound_poly):
                roi_artifacts.append(anno)

        print(len(roi_artifacts))
        mask = np.zeros(arr.shape[:-1])

        points = [[x, y] for x, y in zip(*roi.get_polygon_for_level(
            PyramidalLevel.ZERO, offset=(x_min, y_min)).boundary.coords.xy)]
        print(points)
        mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=1)

        for anno in roi_artifacts:
            geo = anno.get_resized_geometry(1, (x_min, y_min))
            # print(geo)
            points = [[x, y] for x, y in zip(*geo.boundary.coords.xy)]
            mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=0)

        h, w = mask.shape
        data_dict = {0: mask.astype(bool)}
        for i in [2, 4]:
            new_h, new_w = int(h / 2 ** i), int(w / 2 ** i)
            resized_mask = cv2.resize(mask, dsize=(new_w, new_h),
                                      interpolation=cv2.INTER_NEAREST)
            data_dict[i] = resized_mask.astype(bool)

        zarr_image.create_multilevel_group("liver_mask", roi_no, data_dict)

        image_4, _ = zarr_image.rois[0].get_by_level(PyramidalLevel.FOUR)
        mask = dask.array.from_zarr(zarr_image.data.get("liver_mask/0/4"))

        to_plot = image_4.compute()
        to_plot[~mask.compute()] = [255, 255, 255]

        plot_rgb(to_plot, False)

        # mask.astype(bool)
        # plot_pic(mask[::8, ::8])
        # print(mask.shape)
