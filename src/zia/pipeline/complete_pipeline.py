"""Run steps on single image."""

import time

from zia.pipeline.file_management.file_management import SlideFileManager
from zia.pipeline.pipeline_components.pipeline import Pipeline
from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.roi_extraction_component import RoiExtractionComponent
from zia.log import get_logger
from zia.pipeline.pipeline_components.roi_registration_component import SlideRegistrationComponent
from zia.pipeline.pipeline_components.stain_separation_component import StainSeparationComponent, Stain

logger = get_logger(__name__)

if __name__ == "__main__":

    config = get_project_config("steatosis")

    file_manager = SlideFileManager(config.data_path, config.extension)

    pipeline = Pipeline(
        components=[
            # finds ROI of liver tissue
            # RoiDetectionComponent(config, file_manager, overwrite=False),
            # writing rois to ome.tiff file for registration
            # RoiExtractionComponent(config, file_manager, overwrite=False)
            #  valis slide registration
            # SlideRegistrationComponent(config, file_manager, overwrite=False)
            # stain separation
            StainSeparationComponent(config, file_manager, stains=[Stain.ZERO, Stain.ONE], overwrite=False)
        ]
    )

    start_time = time.time()
    pipeline.run()
    end_time = time.time()
    t = (end_time - start_time) / 60
    logger.info(f"Pipeline finished in {t:.2f} min.")
