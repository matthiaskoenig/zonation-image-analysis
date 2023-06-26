"""Pipeline definitions."""

from abc import ABC, abstractmethod

from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.path_utils import FileManager


class IPipelineComponent(ABC):
    """Baseclass for pipeline step."""

    def __init__(self, data_repository: DataRepository, overwrite: bool = False):

        self.data_repository: DataRepository = data_repository
        self.file_manager: FileManager = data_repository.file_manager
        self.overwrite = overwrite

    @abstractmethod
    def run(self) -> None:
        pass
