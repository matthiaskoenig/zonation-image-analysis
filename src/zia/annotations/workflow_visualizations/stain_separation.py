import os

import cv2
import numpy as np

from zia import OPENSLIDE_PATH
from zia.annotations.open_slide_image.data_repository import DataRepository
from zia.annotations.path_utils import FileManager
from zia.annotations.pipeline.stain_separation.macenko import normalize_staining, calculate_optical_density
from zia.annotations.workflow_visualizations.util.image_plotting import (
    plot_pic,
    plot_rgb,
)

if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        pass
else:
    pass


def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contour[:, 0, :]])


def filter_shapes(contours):
    return [contour for contour in contours if contour.shape[0] >= 4]


if __name__ == "__main__":
    from zia import DATA_PATH, REPORT_PATH, RESULTS_PATH, ZARR_PATH

    image_name = "MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006"
    # manages the paths
    file_manager = FileManager(
        data_path=DATA_PATH,
        zarr_path=ZARR_PATH,
        results_path=RESULTS_PATH,
        report_path=REPORT_PATH,
    )

    data_repository = DataRepository(file_manager)

    image = data_repository.image_data_stores.get(image_name).image

    w, h = image.dimensions

    level = 3
    factor = image.level_downsamples[level]

    region = image.read_region(
        location=(100 * 128, 400 * 128), level=level, size=(512, 512)
    )

    image_array = np.array(region)[:, :, :-1]
    print(image_array.shape)
    #plot_rgb(image_array, False)

    od = calculate_optical_density(image_array)
    od = od.reshape((image_array.shape[0], image_array.shape[1], 3))
    print(od.shape)
    print(np.any(od < 0.15, axis=2))
    image_array[np.any(od < 0.15, axis=2)] = (255, 255, 255)

    plot_rgb(image_array, False)
    RC1, RC1N, RC2, RC2N = normalize_staining(
        image_array, Io=240, alpha=1, beta=0.15
    )

    print(RC2.shape)
    plot_pic(RC2)
    plot_pic(RC1)
    # plot_rgb(INorm, transform_to_bgr=False)
