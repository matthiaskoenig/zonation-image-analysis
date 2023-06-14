import os.path
from typing import Optional, List, Tuple

import cv2
from dask.array import from_zarr
import numpy as np
import shapely
from shapely import Polygon

from zia import DATA_PATH, RESULTS_PATH
from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType, \
    Annotation
from zia.annotations.annotation.roi import Roi, PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_rgb
from zia.annotations.zarr_image.zarr_image import ZarrImage

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


class MaskGenerator:

    @classmethod
    def create_mask(cls, zarr_image: ZarrImage) -> None:

        # parse the annotations and filter for artifacts
        annotations = AnnotationParser.get_annotation_by_types(
            AnnotationParser.parse_geojson(get_annotation_path(zarr_image.name)),
            AnnotationType.get_artifacts())

        # iterate over the range of interests
        for roi_no, (leveled_roi, roi) in enumerate(
            zip(zarr_image.rois, zarr_image.roi_annos)):
            (arr, bounds) = leveled_roi.get_by_level(
                PyramidalLevel.ZERO)

            # filter artifact annotations that are located within this roi
            artifacts = MaskGenerator._filter_artifact_annos_for_roi(bounds,
                                                                     annotations)

            # create initial mask array
            mask = np.zeros(arr.shape[:-1])

            offset = (bounds[0], bounds[1])
            # draw polygon for roi
            mask = MaskGenerator._draw_roi_polygon(roi, mask, offset)
            mask = MaskGenerator._draw_artifact_polygons(artifacts, mask, offset)

            # generate lower resolution mask arrays and save as zarr
            MaskGenerator._write_mask_to_zarr(zarr_image, mask, roi_no)

    @classmethod
    def _is_artifact_in_roi(cls, bound_poly: Polygon, anno: Annotation) -> bool:
        return anno.geometry.within(bound_poly) | anno.geometry.intersects(bound_poly)

    @classmethod
    def _filter_artifact_annos_for_roi(cls, bounds: Tuple[int, int, int, int],
                                       annotations: List[Annotation]) -> List[
        Annotation]:
        bound_poly = shapely.box(*bounds)
        return [anno for anno in annotations if
                MaskGenerator._is_artifact_in_roi(bound_poly, anno)]

    @classmethod
    def _draw_roi_polygon(cls, roi: Roi,
                          mask: np.ndarray,
                          offset: Tuple[int, int]) -> np.ndarray:
        geo = roi.get_polygon_for_level(PyramidalLevel.ZERO, offset=offset)
        return MaskGenerator._draw_polygon(geo, mask, 1)

    @classmethod
    def _draw_artifact_polygons(cls, artifacts: List[Annotation],
                                mask: np.ndarray,
                                offset: Tuple[int, int]) -> np.ndarray:
        for anno in artifacts:
            geo = anno.get_resized_geometry(1, offset)
            mask = MaskGenerator._draw_polygon(geo, mask, 0)
        return mask

    @classmethod
    def _draw_polygon(cls, geometry: shapely.Polygon,
                      mask: np.ndarray,
                      color: int) -> np.ndarray:
        points = [[x, y] for x, y in zip(*geometry.boundary.coords.xy)]
        return cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=color)

    @classmethod
    def _write_mask_to_zarr(cls, zarr_image: ZarrImage,
                            mask: np.ndarray,
                            roi_no: int) -> None:
        h, w = mask.shape
        data_dict = {0: mask.astype(bool)}
        for i in [2, 4]:
            new_h, new_w = int(h / 2 ** i), int(w / 2 ** i)
            resized_mask = cv2.resize(mask,
                                      dsize=(new_w, new_h),
                                      interpolation=cv2.INTER_NEAREST)
            data_dict[i] = resized_mask.astype(bool)

        zarr_image.create_multilevel_group("liver_mask", roi_no, data_dict)


if __name__ == "__main__":
    rois = Roi.load_from_file(
        os.path.join(ROI_ANNOTATION_PATH, f"{IMAGE_NAME}.geojson"))

    zarr_image = ZarrImage(IMAGE_NAME, rois)

    ## reading image with tiffile as zarr store

    MaskGenerator.create_mask(zarr_image)

    image_4, _ = zarr_image.rois[0].get_by_level(PyramidalLevel.FOUR)
    mask = from_zarr(zarr_image.data.get("liver_mask/0/4"))

    to_plot = image_4.compute()
    to_plot[~mask.compute()] = [255, 255, 255]

    plot_rgb(to_plot, False)
