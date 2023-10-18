"""Pipeline definitions."""
import time
from pathlib import Path
from typing import List, Type, Optional

from abc import ABC, abstractmethod
from zia.data_store import DataStore
from zia.log import get_logger

logger = get_logger(__name__)


class IPipelineComponent(ABC):
    """Baseclass for pipeline step."""

    def __init__(self, overwrite: bool = False):
        """Component initialization in pipeline.

        :param: overwrite results in zarr store.
        """
        self.overwrite = overwrite

    @abstractmethod
    def run(self, data_store: DataStore, results_path: Path) -> None:
        pass


class Pipeline:
    """Class for managing pipeline steps"""

    def __init__(self, components: List[IPipelineComponent]):
        self.components: List[IPipelineComponent] = components

    def run(self, data_store: DataStore, results_path: Path,
            slice: slice = None) -> None:
        components_to_run = self.components
        if slice:
            components_to_run = components_to_run[slice]

        for pipeline_component in components_to_run:
            t_s = time.time()
            # try :
            pipeline_component.run(data_store=data_store, results_path=results_path)
            # except IndexError:
            #   logger.error(f"Index error occured for {data_store.image_info.metadata.image_id}")
            t_e = time.time()
            t = (t_e - t_s) / 60
            logger.info(f"{pipeline_component.__class__.__name__} finished in {t:.2f} min.")
