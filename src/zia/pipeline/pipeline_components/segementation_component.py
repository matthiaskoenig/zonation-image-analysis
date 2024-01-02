from pathlib import Path
from typing import List, Generator, Tuple, Dict

import cv2

from zia.log import get_logger
from zia.pipeline.common.resolution_levels import PyramidalLevel
from zia.pipeline.file_management.file_management import SlideFileManager, Slide
from zia.pipeline.pipeline_components.pipeline import IPipelineComponent
from zia.pipeline.pipeline_components.stain_separation_component import Stain, StainSeparationComponent
from zia.pipeline.common.project_config import Configuration
from zia.pipeline.pipeline_components.algorithm.segementation.clustering import run_skeletize_image
from zia.pipeline.pipeline_components.algorithm.segementation.filtering import Filter
from zia.pipeline.pipeline_components.algorithm.segementation.get_segments import segment_thinned_image
from zia.pipeline.pipeline_components.algorithm.segementation.load_image_stack import load_image_stack_from_zarr
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats
from zia.pipeline.pipeline_components.algorithm.segementation.process_segment import process_line_segments

logger = get_logger(__file__)


class SegmentationComponent(IPipelineComponent):
    """Pipeline step for lobuli segmentation."""
    dir_name = "LobuliSegmentation"

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, overwrite: bool = False, report: bool = False):
        super().__init__(project_config, file_manager, SegmentationComponent.dir_name, overwrite)
        self.report = report

    def get_lobe_paths(self, subject: str) -> Generator[Tuple[str, Path], None, None]:
        subject_path = self.project_config.image_data_path / StainSeparationComponent.dir_name / f"{Stain.ONE.value}" / subject
        if not subject_path.exists():
            raise FileNotFoundError(f"The stain separation directory for the {Stain.ONE.value} for subject {subject} does not exist.")
        return ((p.name, p) for p in subject_path.iterdir())

    def get_zarr_paths(self, lobe_path: Path, slides: List[Slide]) -> Dict[str, Path]:
        protein_slide_paths = {}
        for slide in slides:
            p = lobe_path / f"{slide.protein}.zarr"
            if not p.exists():
                logger.warning(f"The stain separated image {p} does not exist.")
            else:
                protein_slide_paths[slide.protein] = p
        return protein_slide_paths

    def create_result_path(self, subject: str, lobe_id: str) -> Path:
        p = self.image_data_path / subject / lobe_id
        p.mkdir(exist_ok=True, parents=True)
        return p

    def create_report_path(self, subject: str, lobe_id: str) -> Path:
        p = self.report_path / subject / lobe_id
        p.mkdir(exist_ok=True, parents=True)
        return p

    def check_exists(self, result_path: Path, subject: str, lobe_id: str) -> bool:
        if not self.overwrite and any(result_path.iterdir()):
            logger.info(f"SlideStats for subject {subject} and lobe {lobe_id} already exist. Run with overwrite=True to overwrite")
            return True
        return False

    def run(self) -> None:
        for subject, slides in self.file_manager.group_by_subject().items():
            for lobe_id, lobe_path in self.get_lobe_paths(subject):
                report_path = self.create_report_path(subject, lobe_id)
                result_path = self.create_result_path(subject, lobe_id)

                if self.check_exists(result_path, subject, lobe_id):
                    continue

                protein_slide_paths = self.get_zarr_paths(lobe_path, slides)

                logger.info(f"Started segmentation for {subject}.")
                slide_stats = self.find_lobules_for_subject(protein_slide_paths, report_path)
                slide_stats.meta_data.update(dict(subject=subject, lobe_id=lobe_id))
                slide_stats.to_geojson(result_path)

    def find_lobules_for_subject(self, protein_slide_paths: Dict[str, Path],
                                 report_path: Path = None, pad=10) -> SlideStats:

        loaded_level = PyramidalLevel.FIVE

        logger.info(f"Load image stack for level {loaded_level.value}")

        logger.info("Load images as stack")
        image_stack = load_image_stack_from_zarr(protein_slide_paths, loaded_level)

        logger.info("Applying filters and preprocessing.")
        image_filter = Filter(image_stack, loaded_level)
        final_level, filtered_image_stack = image_filter.prepare_image()

        if self.report:
            for i in range(image_stack.shape[2]):
                cv2.imwrite(str(report_path / f"slide_{i}.png"), filtered_image_stack[:, :, i])

        logger.info("Run superpixel algorithm.")
        thinned, (vessel_classes, vessel_contours) = run_skeletize_image(filtered_image_stack,
                                                                         n_clusters=3,
                                                                         pad=pad,
                                                                         report_path=report_path if self.report else None)

        logger.info("Segmenting lines in thinned image.")
        line_segments = segment_thinned_image(thinned)

        logger.info("Creating lobule and vessel polygons from line segments and vessel contours.")
        slide_stats = process_line_segments(line_segments,
                                            vessel_classes,
                                            vessel_contours,
                                            final_level,
                                            pad)

        if self.report:
            slide_stats.plot(report_path=report_path)

        return slide_stats
