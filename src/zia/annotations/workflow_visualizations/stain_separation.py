import numpy as np

from zia.annotations.pipelines.stain_separation.macenko import (
    calculate_optical_density,
    normalize_staining,
)
from zia.annotations.workflow_visualizations.util.image_plotting import (
    plot_pic,
    plot_rgb,
)
from zia.config import read_config
from zia.data_store import DataStore
from zia.path_utils import FileManager


def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contour[:, 0, :]])


def filter_shapes(contours):
    return [contour for contour in contours if contour.shape[0] >= 4]


if __name__ == "__main__":
    from zia import BASE_PATH

    image_name = (
        "MNT-025_Bl6J_J-20-0160_CYP2E1- 1 400_Run 11_LLL, RML, RSL, ICL_MAA_0006"
    )

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=None
    )

    image_infos = file_manager.get_images()

    image_infos = list(filter(lambda x: x.path.stem == image_name, image_infos))
    image_info = image_infos[0]

    data_store = DataStore(image_info)

    image = data_store.image

    level = 3
    factor = image.level_downsamples[level]

    region = image.read_region(
        location=(100 * 128, 400 * 128), level=level, size=(512, 512)
    )

    image_array = np.array(region)[:, :, :-1]
    print(image_array.shape)
    # plot_rgb(image_array, False)

    od = calculate_optical_density(image_array)
    od = od.reshape((image_array.shape[0], image_array.shape[1], 3))
    print(od.shape)
    print(np.any(od < 0.15, axis=2))
    image_array[np.any(od < 0.15, axis=2)] = (255, 255, 255)

    plot_rgb(image_array, False)
    RC1, RC1N, RC2, RC2N = normalize_staining(image_array, Io=240, alpha=1, beta=0.15)

    print(RC2.shape)
    plot_pic(RC2)
    plot_pic(RC1)
    # plot_rgb(INorm, transform_to_bgr=False)
