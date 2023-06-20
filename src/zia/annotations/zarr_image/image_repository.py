from typing import Dict

from zia.annotations.path_utils.path_util import FileManager
from zia.annotations.zarr_image.zarr_image import ZarrImage


class ImageRepository:
    _zarr_images: Dict[str, ZarrImage] = {}

    @property
    def zarr_images(self) -> Dict[str, ZarrImage]:
        if not self._zarr_images:
            self._initialize()
        return self._zarr_images

    def __init__(self, file_manager: FileManager):
        self._file_manager = file_manager

    def _initialize(self):
        self._zarr_images = {
            name: ZarrImage(name, self._file_manager)
            for name, name in self._file_manager.get_image_names()
        }
