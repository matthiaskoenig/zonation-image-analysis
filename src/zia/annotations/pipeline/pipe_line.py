from typing import List, Type

from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.path_utils import FileManager
from zia.annotations.pipeline.abstract_pipeline.pipeline import IPipelineComponent
from zia.annotations.pipeline.liver_roi_detection.pipeline_component import (
    RoiFinderComponent,
)
from zia.annotations.pipeline.mask_generatation.pipeline_component import (
    MaskCreationComponent,
)


class Pipeline:
    def __init__(self, data_repository: DataRepository):
        self.data_repository = data_repository
        self.pipeline_components: List[IPipelineComponent] = []

    def register_component(
        self,
        pipeline_component_class: Type[IPipelineComponent],
        overwrite: bool = False,
    ) -> None:
        self.pipeline_components.append(
            pipeline_component_class(self.data_repository, overwrite)
        )

    def run(self, slice: slice = None) -> None:
        components_to_run = self.pipeline_components
        if slice:
            components_to_run = components_to_run[slice]

        for pipeline_component in components_to_run:
            pipeline_component.run()
