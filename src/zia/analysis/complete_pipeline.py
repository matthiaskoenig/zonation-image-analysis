from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.path_utils import FileManager
from zia.annotations.pipeline.liver_roi_detection.pipeline_component import (
    RoiFinderComponent,
)
from zia.annotations.pipeline.mask_generatation.pipeline_component import (
    MaskCreationComponent,
)
from zia.annotations.pipeline.pipe_line import Pipeline


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
    pipeline.register_component(MaskCreationComponent, overwrite=True)

    pipeline.run()
