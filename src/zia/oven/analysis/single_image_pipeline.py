"""Run steps on single image."""
from zia.oven.annotations.pipelines.liver_roi_detection.pipeline_component import (
    RoiFinderComponent,
)
from zia.oven.annotations.pipelines.pipeline import Pipeline
from zia.oven.data_store import DataStore
from zia.oven.path_utils import FileManager, filter_factory

if __name__ == "__main__":
    from zia import BASE_PATH
    from zia.config import read_config

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=filter_factory(species="mouse", subject="MNT-022", protein="cyp2d6")
    )

    pipeline = Pipeline(
        components=[
            # finds ROI of liver tissue
            RoiFinderComponent(overwrite=True, draw=True),
            # creates masks
            #MaskCreationComponent(overwrite=False),
            # separate stains
            #StainSeparationComponent(overwrite=True)
        ]
    )

    for image_info in file_manager.get_images():
        data_store = DataStore(image_info=image_info)
        pipeline.run(data_store, results_path=file_manager.results_path)
