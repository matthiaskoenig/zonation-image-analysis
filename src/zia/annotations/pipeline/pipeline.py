"""Pipeline definitions."""

from abc import ABC, abstractmethod

from zia.annotations.path_utils import FileManager
from zia.annotations.zarr_image.image_repository import ImageRepository


class IPipelineComponent(ABC):
    """Baseclass for pipeline step."""

    def __init__(self, file_manager: FileManager, image_repository: ImageRepository):
        self._file_manager: FileManager = file_manager
        self._image_repository: ImageRepository = image_repository

    @abstractmethod
    def run(self) -> None:
        pass
