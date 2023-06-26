"""Run the image processing pipeline."""

import logging
import time
from typing import List

from PIL.ImageDraw import ImageDraw

from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.open_slide_image.data_store import DataStore
from zia.annotations.path_utils import FileManager, ResultDir
from zia.annotations.pipeline.abstract_pipeline.pipeline import IPipelineComponent
from zia.annotations.pipeline.liver_roi_detection.image_analysis import RoiSegmentation
from zia.annotations.pipeline.liver_roi_detection.report import RoiSegmentationReport
from zia.console import console
from zia.io.wsi_openslide import read_full_image_from_slide

logger = logging.getLogger(__name__)


class RoiFinderComponent(IPipelineComponent):
    """Pipeline step for ROI processing."""
    _reports_dict = {}  # FIXME: not class

    def __init__(self, data_repository: DataRepository, overwrite=False, draw: bool = True):
        IPipelineComponent.__init__(self, data_repository, overwrite)
        self._draw = draw

    def run(self) -> None:
        """Run the analysis/Roi processing."""
        for species, image_name in self.file_manager.get_image_names():
            console.print(species, image_name)
            self._find_rois(species, image_name)

        self._save_reports()

    def _find_rois(self, species: str, image_name: str) -> None:
        report = self._get_report(species)
        start_time = time.time()
        geojson_path = self.file_manager.get_annotation_path(image_name)

        if not geojson_path:
            # logger.warning("No annotation geojson file found for '" + image_name + "'.")
            report.register_geojson_missing(image_name)
            return

        annotations = AnnotationParser.parse_geojson(geojson_path)

        liver_annotations = AnnotationParser.get_annotation_by_type(
            annotations, AnnotationType.LIVER
        )

        if len(liver_annotations) == 0:
            report.register_liver_annotation_missing(image_name)
            return

        data_store = self.data_repository.image_data_stores.get(image_name)

        liver_rois = RoiSegmentation.find_rois(
            data_store, annotations, AnnotationType.LIVER
        )

        if len(liver_rois) == 0:
            logger.warning("No ROI found for '" + image_name + "'.")
            report.register_segmentation_fail(image_name)

        else:
            self._save_rois(liver_rois, species, image_name)
            self._draw_result(data_store, liver_rois, species, image_name)

            if len(liver_rois) == len(liver_annotations):
                report.register_segmentation_success(image_name)
            else:
                report.register_segmentation_partial(image_name)

        end_time = time.time()
        report.set_time(end_time - start_time)

        logger.info(f"Finished finding liver ROIs for {species}")

    def _save_reports(self) -> None:
        for species, report in self._reports_dict.items():
            console.print(report)
            report.save(
                self.file_manager.get_report_path(
                    ResultDir.ANNOTATIONS_LIVER_ROI, species, "report.txt"
                )
            )

    def _get_report(self, species: str) -> RoiSegmentationReport:
        if species not in self._reports_dict.keys():
            self._reports_dict[species] = RoiSegmentationReport()
        return self._reports_dict[species]

    def _save_rois(self, rois: List[Roi], species: str, image_name: str):
        if len(rois) != 0:
            Roi.write_to_geojson(
                rois,
                self.file_manager.get_results_path(
                    ResultDir.ANNOTATIONS_LIVER_ROI, species, f"{image_name}.geojson"
                ),
            )

    def _draw_result(
            self,
            data_store: DataStore,
            liver_rois: List[Roi],
            species: str,
            image_name: str,
    ) -> None:
        if not self._draw:
            return

        region = read_full_image_from_slide(data_store.image, 7)
        draw = ImageDraw(region)
        for liver_roi in liver_rois:
            poly_points = liver_roi.get_polygon_for_level(
                PyramidalLevel.SEVEN
            ).exterior.coords
            draw.polygon(list(poly_points), outline="red", width=3)

        region.save(
            self.file_manager.get_report_path(
                ResultDir.ANNOTATIONS_LIVER_ROI, species, f"{image_name}.png"
            ),
            "PNG",
        )


if __name__ == "__main__":
    from zia import DATA_PATH, REPORT_PATH, RESULTS_PATH, ZARR_PATH

    # manages the paths
    file_manager = FileManager(
        data_path=DATA_PATH,
        zarr_path=ZARR_PATH,
        results_path=RESULTS_PATH,
        report_path=REPORT_PATH,
    )
    # manages the actual image data
    data_repository = DataRepository(file_manager)

    roi_segmentation = RoiFinderComponent(data_repository)
    roi_segmentation.run()
