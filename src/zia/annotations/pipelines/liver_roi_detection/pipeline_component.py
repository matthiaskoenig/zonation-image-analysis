"""Run the image processing pipeline."""

import time
from pathlib import Path
from typing import List

from PIL.ImageDraw import ImageDraw

from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.liver_roi_detection.image_analysis import RoiSegmentation
from zia.annotations.pipelines.pipeline import IPipelineComponent

from zia.path_utils import ResultsDirectories
from zia.data_store import DataStore
from zia.console import console
from zia.io.wsi_openslide import read_full_image_from_slide
from zia.log import get_logger

logger = get_logger(__name__)


class RoiFinderComponent(IPipelineComponent):
    """Pipeline step for ROI processing."""

    _reports_dict = {}  # FIXME: not class

    def __init__(self, overwrite=False, draw: bool = True):
        super().__init__(overwrite)
        self._draw = draw

    def run(self, data_store: DataStore, results_path: Path) -> None:
        """Run analysis."""
        # report = self._get_report(species)
        image_id = data_store.image_info.metadata.image_id

        start_time = time.time()

        geojson_path = data_store.image_info.annotations_path
        if not geojson_path:
            logger.warning(f"No annotation geojson file found for {image_id}.")
            # report.register_geojson_missing(image_name)
            return

        annotations = AnnotationParser.parse_geojson(
            path=geojson_path
        )

        liver_annotations = AnnotationParser.get_annotation_by_type(
            annotations, AnnotationType.LIVER
        )

        if len(liver_annotations) == 0:
            # report.register_liver_annotation_missing(image_name)
            return

        liver_rois = RoiSegmentation.find_rois(
            data_store, annotations, AnnotationType.LIVER
        )

        if len(liver_rois) > 0:
            roi_path = results_path / ResultsDirectories.ANNOTATIONS_LIVER_ROI / f"{image_id}.geojson"
            Roi.write_to_geojson(
                rois=liver_rois,
                path=roi_path,
            )
            # update roi path
            data_store.image_info.roi_path = roi_path

            if self._draw:
                image_path = results_path / ResultsDirectories.ANNOTATIONS_LIVER_ROI / f"{image_id}.png"
                self._draw_rois(
                    liver_rois=liver_rois,
                    image_path=image_path,
                    data_store=data_store,
                )

        else:
            logger.warning(f"No ROI found for {image_id}")
            # report.register_segmentation_fail(image_name)

            # if len(liver_rois) == len(liver_annotations):
            #     report.register_segmentation_success(image_name)
            # else:
            #     report.register_segmentation_partial(image_name)

        end_time = time.time()
        # report.set_time(end_time - start_time)

        logger.info(f"Finished finding liver ROIs for {image_id}")
        # self._save_reports()

    @staticmethod
    def _draw_rois(
        liver_rois: List[Roi],
        image_path: Path,
        data_store: DataStore,
    ) -> None:
        """Draw rois."""
        region = read_full_image_from_slide(data_store.image, 7)
        draw = ImageDraw(region)
        for liver_roi in liver_rois:
            poly_points = liver_roi.get_polygon_for_level(
                PyramidalLevel.SEVEN
            ).exterior.coords
            draw.polygon(list(poly_points), outline="red", width=3)

        region.save(image_path, "PNG")

    #
    # def _save_reports(self) -> None:
    #     for species, report in self._reports_dict.items():
    #         console.print(report)
    #         report.save(
    #             self.file_manager.get_report_path(
    #                 ResultsDirectories.ANNOTATIONS_LIVER_ROI, species, "report.txt"
    #             )
    #         )
    #
    # def _get_report(self, species: str) -> RoiSegmentationReport:
    #     if species not in self._reports_dict.keys():
    #         self._reports_dict[species] = RoiSegmentationReport()
    #     return self._reports_dict[species]
