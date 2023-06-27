"""Run steps on single image."""

from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.path_utils import FileManager
from zia.annotations.pipeline.liver_roi_detection.pipeline_component import (
    RoiFinderComponent,
)
from zia.annotations.pipeline.mask_generatation.pipeline_component import (
    MaskCreationComponent,
)
from zia.annotations.pipeline.pipe_line import Pipeline
from zia.console import console


if __name__ == "__main__":
    from zia import BASE_PATH
    from zia.config import read_config

    configuration = read_config(BASE_PATH / "configuration.ini")

    # manages the paths
    file_manager = FileManager(configuration)
    data_repository = DataRepository(file_manager)
    pipeline = Pipeline(data_repository)
    pipeline.register_component(RoiFinderComponent)
    pipeline.register_component(MaskCreationComponent, overwrite=True)

    pipeline.run()
