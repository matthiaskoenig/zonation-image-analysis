"""Pipeline definitions."""

from abc import ABC, abstractmethod

from zia.annotations.path_utils import FileManager
from zia.annotations.zarr_image.image_repository import ImageRepository


class IPipelineComponent(ABC):
    """Baseclass for pipeline step."""

    def __init__(self, image_repository: ImageRepository):

        self.image_repository: ImageRepository = image_repository
        self.file_manager: FileManager = image_repository.file_manager

    @abstractmethod
    def run(self) -> None:
        pass
