import numpy as np
from PIL import Image

from zia import DATA_PATH, REPORT_PATH, RESULTS_PATH, ZARR_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.open_slide_image.data_store import ZarrGroups
from zia.annotations.path_utils import FileManager, ResultsDirectories
from zia.annotations.workflow_visualizations.util.image_plotting import plot_rgb


if __name__ == "__main__":
    # manages the paths
    file_manager = FileManager(
        data_path=DATA_PATH,
        zarr_path=ZARR_PATH,
        results_path=RESULTS_PATH,
        report_path=REPORT_PATH,
    )

    data_repository = DataRepository(file_manager)

    for species, name in file_manager.image_paths():
        data_store = data_repository.image_data_stores.get(name)

        for i, roi in enumerate(data_store.rois):
            mask_array = data_store.data.get(f"{ZarrGroups.LIVER_MASK.value}/{i}/{0}")

            down_sample_array = mask_array[::16, ::16]
            print(down_sample_array.shape)
            xs_ref, ys_ref = roi.get_bound(PyramidalLevel.ZERO)
            ref_loc = xs_ref.start, ys_ref.start

            xs, ys = roi.get_bound(PyramidalLevel.FOUR)

            size = (xs.stop - xs.start, ys.stop - ys.start)

            roi_image = data_store.image.read_region(
                location=ref_loc, level=PyramidalLevel.FOUR, size=size
            )

            image_array = np.array(roi_image)
            print(image_array.shape)

            image_array[~down_sample_array] = (255, 255, 255, 0)

            masked_image = Image.fromarray(image_array)

            plot_rgb(masked_image, transform_to_bgr=False)

            masked_image.save(
                file_manager.get_report_path(
                    ResultsDirectories.MASKED_PNG_IMAGES, species, f"{name}_{i}.png"
                )
            )
