from typing import Dict

from zia.annotations.open_slide_image.data_store import DataStore
from zia.annotations.path_utils import FileManager


class DataRepository:
    """
    Class that holds the Data Stores for each image in a dict with name as key.
    """

    def __init__(self, file_manager: FileManager):
        self.file_manager: FileManager = file_manager
        self._image_data_stores: Dict[str, DataStore] = {}

    @property
    def image_data_stores(self) -> Dict[str, DataStore]:
        """Get OpenSlides images."""
        if not self._image_data_stores:
            self._image_data_stores = {
                name: DataStore(name, file_manager=self.file_manager)
                for _, name in self.file_manager.get_image_names()
            }

        return self._image_data_stores
