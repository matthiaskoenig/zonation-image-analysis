from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.open_slide_image.data_store import DataStore, ZarrGroups
from zia.annotations.path_utils import FileManager
from zia.annotations.pipeline.abstract_pipeline.pipeline import IPipelineComponent
from zia.annotations.pipeline.stain_separation.image_analysis import StainSeparator


class StainSeparationPipelineComponent(IPipelineComponent):
    def __init__(self, data_repository: DataRepository, overwrite: bool = False):
        IPipelineComponent.__init__(self, data_repository, overwrite)

    def run(self) -> None:
        for species, image_name in self.file_manager.get_image_names():
            # filter cyp images
            if not "CYP" in image_name:
                continue
            if "Negative_Run" in image_name:
                return

            print(species, image_name)
            data_store = self.data_repository.image_data_stores.get(image_name)

            # prevent from overwriting data from previous runs during development
            if self._check_if_exists(data_store) & ~self.overwrite:
                continue

            StainSeparator.separate_stains(data_store)
            break

    @classmethod
    def _check_if_exists(cls, data_store: DataStore) -> bool:
        if ZarrGroups.DAB_STAIN.value in data_store.data.keys():
            return True
        else:
            return False


if __name__ == "__main__":
    from zia import DATA_PATH, REPORT_PATH, RESULTS_PATH, ZARR_PATH

    # manages the paths
    file_manager = FileManager(
        data_path=DATA_PATH,
        zarr_path=ZARR_PATH,
        results_path=RESULTS_PATH,
        report_path=REPORT_PATH,
    )

    data_repository = DataRepository(file_manager)
    stain_separator = StainSeparationPipelineComponent(data_repository, False)
    stain_separator.run()
