"""Image repository.

Storage for zarr images.
"""
from typing import Dict

from zia.annotations.path_utils import FileManager
from zia.annotations.zarr_image.zarr_image import ZarrImage


class ImageRepository:
    """Class for handling images.

    Images correspond to the images provided by a FileManager.
    """

    def __init__(self, file_manager: FileManager):
        self.file_manager: FileManager = file_manager
        self._zarr_images: Dict[str, ZarrImage] = {}

    @property
    def zarr_images(self) -> Dict[str, ZarrImage]:
        """Get zarr images."""
        if not self._zarr_images:
            self._zarr_images = {
                name: ZarrImage(name, self.file_manager)
                for name, name in self.file_manager.get_image_names()
            }

        return self._zarr_images
