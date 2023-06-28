"""Run steps on single image."""

from zia.annotations.pipelines.liver_roi_detection.pipeline_component import (
    RoiFinderComponent,
)
from zia.annotations.pipelines.mask_generatation.pipeline_component import (
    MaskCreationComponent,
)
from zia.annotations.pipelines.pipeline import Pipeline
from zia.data_store import DataStore
from zia.path_utils import FileManager

if __name__ == "__main__":
    from zia import BASE_PATH
    from zia.config import read_config

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=None
    )

    pipeline = Pipeline(
        components=[
            # finds ROI of liver tissue
            RoiFinderComponent(overwrite=False),
            # creates masks
            MaskCreationComponent(overwrite=False),
        ]
    )

    for image_info in file_manager.get_images():
        data_store = DataStore(image_info=image_info)
        pipeline.run(data_store, results_path=file_manager.results_path)
