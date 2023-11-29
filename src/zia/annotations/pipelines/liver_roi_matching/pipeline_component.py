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
from zia.data_store import DataStore
from zia.io.wsi_openslide import read_full_image_from_slide
from zia.log import get_logger, create_message
from zia.path_utils import ResultsDirectories

logger = get_logger(__name__)


class RoiFinderComponent(IPipelineComponent):
    """Pipeline step for ROI processing."""

    def __init__(self, overwrite=False, draw: bool = True):
        super().__init__(overwrite)
        self._draw = draw

    def run(self, data_store: DataStore, results_path: Path) -> None:
        """Run analysis."""
        image_id = data_store.image_info.metadata.image_id

        # prevent from overwriting existing data
        if not self.overwrite and data_store.image_info.roi_path.exists():
            logger.info(create_message(image_id,
                                       "The ROI file already exists. Set overwrite flag"
                                       " to True to allow overwriting."))
            return

        logger.info(f"[{image_id}]\tStarted ROI detection.")
        start_time = time.time()

        geojson_path = data_store.image_info.annotations_path

        if not geojson_path:
            logger.warning(f"[{image_id}]\tNo annotation geojson file found.")
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

        # writing this as a subdirectory to the zarr store because this is reused.
        if len(liver_rois) > 0:
            # adding data to the data_store.
            data_store.register_rois(liver_rois)

            if self._draw:
                image_path = results_path / ResultsDirectories.LIVER_ROIS_IMAGES.value
                image_path.mkdir(exist_ok=True)

                self._draw_rois(
                    liver_rois=liver_rois,
                    image_path=image_path / f"{image_id}.png",
                    data_store=data_store,
                )

        else:
            logger.warning(f"[{image_id}]\tNo ROIs found.")

        end_time = time.time()
        run_time = end_time - start_time

        logger.info(
            f"[{image_id}]\tFinished ROI detection in {run_time:.2f} s. Found {len(liver_rois)} ROI.")
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
