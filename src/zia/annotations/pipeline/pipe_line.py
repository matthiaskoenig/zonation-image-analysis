from typing import List, Type

from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.path_utils import FileManager
from zia.annotations.pipeline.abstract_pipeline.pipeline import IPipelineComponent
from zia.annotations.pipeline.liver_roi_detection.pipeline_component import \
    RoiFinderComponent
from zia.annotations.pipeline.mask_generatation.pipeline_component import \
    MaskCreationComponent


class Pipeline:

    def __init__(self, data_repository: DataRepository):
        self.data_repository = data_repository
        self.pipeline_components: List[IPipelineComponent] = []

    def register_component(self, pipeline_component_class: Type[IPipelineComponent]) -> None:
        self.pipeline_components.append(pipeline_component_class(self.data_repository))

    def run(self, slice: slice = None) -> None:
        components_to_run = self.pipeline_components
        if slice:
            components_to_run = components_to_run[slice]

        for pipeline_component in components_to_run:
            pipeline_component.run()


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

    pipeline = Pipeline(data_repository)

    pipeline.register_component(RoiFinderComponent)
    pipeline.register_component(MaskCreationComponent)

    pipeline.run()

