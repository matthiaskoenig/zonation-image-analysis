from pathlib import Path

from zia.data_store import DataStore, ZarrGroups
from zia.log import get_logger, create_message
from zia.path_utils import FileManager
from zia.annotations.pipelines.pipeline import IPipelineComponent
from zia.annotations.pipelines.stain_separation.image_analysis import separate_stains

logger = get_logger(__name__)


class StainSeparationComponent(IPipelineComponent):
    def __init__(self, overwrite: bool = False):
        IPipelineComponent.__init__(self, overwrite)

    def run(self, data_store: DataStore, results_path: Path) -> None:
        image_id = data_store.image_info.metadata.image_id

        if data_store.image_info.metadata.negative:
            logger.info("Skipped Stain separation for negative run image.")
            return

        logger.info(create_message(image_id, "Started stain separation."))

        # prevent from overwriting data from previous runs during development
        if self._check_if_exists(data_store) & ~self.overwrite:
            logger.info(
                f"[{image_id}]\t Spearated image already exists. To overwrite, set overwrite to True for {self.__class__.__name__}.")
            return

        separate_stains(data_store)

        logger.info(create_message(image_id, "Finished Stain Separation"))

    @classmethod
    def _check_if_exists(cls, data_store: DataStore) -> bool:
        if (ZarrGroups.STAIN_0.value in data_store.data.keys()) or (ZarrGroups.STAIN_1 in data_store.data.keys()):
            return True
        else:
            return False
