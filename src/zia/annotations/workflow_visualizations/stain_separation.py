import cv2
import numpy as np
from matplotlib import pyplot as plt

from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.stain_separation.macenko import (
    calculate_optical_density,
    normalize_staining, normalizeStaining,
)
from zia.annotations.workflow_visualizations.util.image_plotting import (
    plot_pic,
    plot_rgb,
)
from zia.config import read_config
from zia.data_store import DataStore
from zia.io.wsi_openslide import read_full_image_from_slide
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

    region = data_store.read_full_roi(0, PyramidalLevel.THREE)

    image_array = np.array(region)

    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    gs = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    threshold, _ = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #print(image_array.shape)
    # plot_rgb(image_array, False)

    #plot_rgb(image_array, False)
    Io, HE, DAB = normalizeStaining(image_array, gs_threshold=threshold, Io=240, alpha=1)

    histogram = cv2.calcHist([DAB], [0], None, [256], [0, 256])

    plt.plot(histogram, color='black')


    plot_rgb(Io, False)
    plot_pic(HE, "Hematoxylin")
    plot_pic(DAB, "DAB")
    # plot_rgb(INorm, transform_to_bgr=False)
