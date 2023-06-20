import logging
import os
import time
from typing import List

from PIL.ImageDraw import ImageDraw

from zia.annotations import OPENSLIDE_PATH
from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.annotation.geometry_utils import read_full_image_from_slide
from zia.annotations.annotation.roi import Roi, PyramidalLevel
from zia.annotations.path_utils.path_util import FileManager, ResultDir
from zia.annotations.pipeline.liver_roi_detection.image_analysis import RoiSegmentation
from zia.annotations.pipeline.liver_roi_detection.report import RoiSegmentationReport
from zia.annotations.pipeline.pipeline import IPipelineComponent
from zia.annotations.zarr_image.image_repository import ImageRepository
from zia.annotations.zarr_image.zarr_image import ZarrImage

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

logger = logging.getLogger(__name__)


class RoiFinderComponent(IPipelineComponent):
    _reports_dict = {}

    def __init__(self, file_manager: FileManager, image_repo: ImageRepository, draw: bool = True):
        IPipelineComponent.__init__(self, file_manager, image_repo)
        self._draw = draw

    def run(self) -> None:
        for species, image_name in self._file_manager.get_image_names():
            print(species, image_name)
            self._find_rois(species, image_name)
        self._save_reports()

    def _find_rois(self, species: str, image_name: str) -> None:
        report = self._get_report(species)
        start_time = time.time()
        geojson_path = self._file_manager.get_annotation_path(image_name)

        if not geojson_path:
            # logger.warning("No annotation geojson file found for '" + image_name + "'.")
            report.register_geojson_missing(image_name)
            return

        annotations = AnnotationParser.parse_geojson(geojson_path)

        liver_annotations = AnnotationParser.get_annotation_by_type(
            annotations,
            AnnotationType.LIVER)

        if len(liver_annotations) == 0:
            report.register_liver_annotation_missing(image_name)
            return

        open_slide = openslide.OpenSlide(self._file_manager.get_image_path(image_name))

        liver_rois = RoiSegmentation.find_rois(open_slide, annotations,
                                               AnnotationType.LIVER)

        if len(liver_rois) == 0:
            logger.warning("No ROI found for '" + image_name + "'.")
            report.register_segmentation_fail(image_name)

        else:
            self._save_rois(liver_rois, species, image_name)
            self._draw_result(open_slide, liver_rois, species, image_name)

            if len(liver_rois) == len(liver_annotations):
                report.register_segmentation_success(image_name)
            else:
                report.register_segmentation_partial(image_name)

        end_time = time.time()
        report.set_time(end_time - start_time)

        logger.info(f"Finished finding liver ROIs for {species}")

    def _save_reports(self) -> None:
        for species, report in self._reports_dict.items():
            print(report)
            report.save(
                self._file_manager.get_report_path(ResultDir.ANNOTATIONS_LIVER_ROI,
                                                   species,
                                                   "report.txt"))

    def _get_report(self, species: str) -> RoiSegmentationReport:
        if species not in self._reports_dict.keys():
            self._reports_dict[species] = RoiSegmentationReport()
        return self._reports_dict[species]

    def _save_rois(self, rois: List[Roi], species: str, image_name: str):
        if len(rois) != 0:
            Roi.write_to_geojson(rois,
                                 self._file_manager.get_results_path(
                                     ResultDir.ANNOTATIONS_LIVER_ROI,
                                     species,
                                     f"{image_name}.geojson"))

    def _draw_result(self, open_slide: openslide.OpenSlide,
                     liver_rois: List[Roi], species: str, image_name: str) -> None:
        if not self._draw:
            return

        region = read_full_image_from_slide(open_slide, 7)
        draw = ImageDraw(region)
        for liver_roi in liver_rois:
            poly_points = liver_roi.get_polygon_for_level(
                PyramidalLevel.SEVEN).exterior.coords
            draw.polygon(list(poly_points), outline="red", width=3)

        region.save(self._file_manager.get_report_path(ResultDir.ANNOTATIONS_LIVER_ROI,
                                                       species,
                                                       f"{image_name}.png"),
                    "PNG")


if __name__ == "__main__":
    file_manager = FileManager()
    image_repo = ImageRepository(file_manager)
    roi_segmentation = RoiFinderComponent(file_manager, image_repo)
    roi_segmentation.run()
