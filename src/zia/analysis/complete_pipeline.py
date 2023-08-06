"""Run steps on single image."""

from zia.annotations.pipelines.liver_roi_detection.pipeline_component import (
    RoiFinderComponent,
)
from zia.annotations.pipelines.mask_generatation.pipeline_component import (
    MaskCreationComponent,
)
from zia.annotations.pipelines.pipeline import Pipeline
from zia.annotations.pipelines.stain_separation.pipeline_component import \
    StainSeparationComponent
from zia.data_store import DataStore
from zia.log import get_logger
from zia.path_utils import FileManager

import time

logger = get_logger(__name__)

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
            RoiFinderComponent(overwrite=False, draw=False),
            # creates masks
            MaskCreationComponent(overwrite=False, draw=False),
            # stain separation
            StainSeparationComponent(overwrite=False)
        ]
    )

    start_time = time.time()
    for image_info in file_manager.get_images():
        data_store = DataStore(image_info=image_info)
        pipeline.run(data_store, results_path=file_manager.results_path)

    end_time = time.time()
    t = (end_time-start_time)/60
    logger.info(
        f"Pipeline finished in {t:.2f} min.")
    # self._save_reports()
