"""Pipeline definitions."""
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from zia.pipeline.file_management.file_management import SlideFileManager
from zia.pipeline.common.project_config import Configuration
from zia.log import get_logger

logger = get_logger(__name__)


class IPipelineComponent(ABC):
    """Baseclass for pipeline step."""

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, dir_name: str, overwrite: bool = False):
        """Component initialization in pipeline.

        :param: overwrite results in zarr store.
        """
        self.overwrite = overwrite
        self.project_config = project_config
        self.report_path = IPipelineComponent.initialize_report_path(project_config, dir_name)
        self.file_manager = file_manager
        self.image_data_path = IPipelineComponent.initialized_image_data_path(project_config, dir_name)

    @abstractmethod
    def run(self) -> None:
        pass

    def get_report_path(self) -> Path:
        return self.report_path

    def get_image_data_path(self) -> Path:
        return self.image_data_path

    @classmethod
    def initialize_report_path(cls, project_config: Configuration, dir_name: str) -> Path:
        p = project_config.reports_path / dir_name
        p.mkdir(exist_ok=True, parents=True)
        return p

    @classmethod
    def initialized_image_data_path(cls, project_config: Configuration, dir_name: str) -> Path:
        p = project_config.image_data_path / dir_name
        p.mkdir(exist_ok=True, parents=True)
        return p


class Pipeline:
    """Class for managing pipeline steps"""

    def __init__(self, components: List[IPipelineComponent]):
        self.components: List[IPipelineComponent] = components

    def run(self) -> None:
        for pipeline_component in self.components:
            t_s = time.time()
            # try :
            pipeline_component.run()

            t_e = time.time()
            t = (t_e - t_s) / 60
            logger.info(f"{pipeline_component.__class__.__name__} finished in {t:.2f} min.")
