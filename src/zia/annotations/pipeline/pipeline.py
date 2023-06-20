from abc import ABC, abstractmethod

from zia.annotations.path_utils.path_util import FileManager
from zia.annotations.zarr_image.image_repository import ImageRepository
from zia.annotations.zarr_image.zarr_image import ZarrImage


class IPipelineComponent(ABC):
    def __init__(self, file_manager: FileManager, image_repositoy: ImageRepository):
        self._file_manager: FileManager = file_manager
        self._image_repo: ImageRepository = image_repositoy

    @abstractmethod
    def run(self):
        pass




