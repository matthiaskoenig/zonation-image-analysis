from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.open_slide_image.data_store import DataStore, ZarrGroups
from zia.annotations.pipeline.abstract_pipeline.pipeline import IPipelineComponent
from zia.annotations.pipeline.stain_separation.image_analysis import StainSeparator


class StainSeparationPipelineComponent(IPipelineComponent):
    def __init__(self, data_repository: DataRepository, overwrite: bool):
        IPipelineComponent.__init__(self, data_repository)
        self._overwrite = overwrite

    def run(self) -> None:
        for species, image_name in self.file_manager.get_image_names():
            print(species, image_name)
            data_store = self.data_repository.image_data_stores.get(image_name)

            # prevent from overwriting data from previous runs during development
            if self._check_if_exists(data_store) & ~self._overwrite:
                continue

            StainSeparator.separate_stains(data_store)


    @classmethod
    def _check_if_exists(cls, data_store: DataStore) -> bool:
        if ZarrGroups.DAB_STAIN.value in data_store.data.keys():
            return True
        else:
            return False
