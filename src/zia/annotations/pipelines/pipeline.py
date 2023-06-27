"""Pipeline definitions."""
from pathlib import Path
from typing import List, Type, Optional

from abc import ABC, abstractmethod
from zia.data_store import DataStore


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

    def run(self, data_store: DataStore, results_path: Path, slice: slice = None) -> None:
        components_to_run = self.components
        if slice:
            components_to_run = components_to_run[slice]

        for pipeline_component in components_to_run:
            pipeline_component.run(data_store=data_store, results_path=results_path)
