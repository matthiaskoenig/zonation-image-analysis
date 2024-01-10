from pathlib import Path
from typing import List, Dict

from zia.io.wsi_tifffile import read_ndpi
from zia.pipeline.common.project_config import Configuration
from zia.pipeline.file_management.file_management import SlideFileManager, Slide
from zia.pipeline.pipeline_components.pipeline import IPipelineComponent
from zia.pipeline.pipeline_components.roi_extraction_component import RoiExtractionComponent


class DropletDetectionPipelineComponent(IPipelineComponent):
    dir_name = "DropletDetection"

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, overwrite: bool = False):
        super().__init__(project_config, file_manager, DropletDetectionPipelineComponent.dir_name, overwrite)

    def _get_roi_dirs(self, slide: Slide) -> Dict[str, Path]:
        p = self.project_config.image_data_path / RoiExtractionComponent.dir_name / slide.subject
        if not p.exists():
            raise FileNotFoundError(f"No roi directory found for subject {slide.subject}.")

        roi_image_paths = {}
        for roi_dir in p.iterdir():
            img_path = roi_dir / f"{slide.name}.{self.project_config.extension}"
            if not img_path.exists():
                raise FileNotFoundError(f"No image found for subject {slide.subject}, roi {roi_dir.stem}.")

            roi_image_paths[roi_dir.stem] = img_path

        return roi_image_paths

    def run(self) -> None:
        for slide in self.file_manager.slides:

            for lobe_id, path in self._get_roi_dirs(slide):

                self.detect_droplets(slide, lobe_id, path)

    def detect_droplets(self, slide: Slide, lobe_id: str, path: Path):

        array = read_ndpi(path)[0]

        detect_droplets(array)




