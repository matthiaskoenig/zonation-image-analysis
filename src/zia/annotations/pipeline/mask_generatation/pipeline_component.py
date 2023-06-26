from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.open_slide_image.data_store import DataStore, ZarrGroups
from zia.annotations.path_utils import FileManager
from zia.annotations.pipeline.mask_generatation.image_analysis import MaskGenerator
from zia.annotations.pipeline.abstract_pipeline.pipeline import IPipelineComponent


class MaskCreationComponent(IPipelineComponent):
    def __init__(self, data_repository: DataRepository, overwrite=False, draw=True):
        IPipelineComponent.__init__(self, data_repository, overwrite)
        self._draw = draw

    def run(self):
        for species, image_name in self.file_manager.get_image_names():
            print(species, image_name)
            data_store = self.data_repository.image_data_stores.get(image_name)

            # prevent from overwriting data from previous runs during development
            if self._check_if_exists(data_store) & ~self.overwrite:
                continue

            annotations = AnnotationParser.parse_geojson(self.file_manager.get_annotation_path(image_name))
            annotations = AnnotationParser.get_annotation_by_types(annotations, AnnotationType.get_artifacts())

            MaskGenerator.create_mask(data_store, annotations)

            # self._draw_mask(zarr_image, species)

    @classmethod
    def _check_if_exists(cls, data_store: DataStore) -> bool:
        if ZarrGroups.LIVER_MASK.value in data_store.data.keys():
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
    mask_generator = MaskCreationComponent(data_repository, False, True)
    mask_generator.run()
