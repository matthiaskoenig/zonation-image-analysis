import os.path

import numpy as np

from zia import DATA_PATH, RESULTS_PATH
from zia.annotations import OPENSLIDE_PATH
from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import (
    plot_pic,
    plot_rgb,
)


if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

IMAGE_NAME = "J-12-00348_NOR-021_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006"

ANNOTATION_PATH = RESULTS_PATH / "annotations_liver_roi" / "rat"
IMAGE_PATH = DATA_PATH / "cyp_species_comparison" / "all" / "rat" / "CYP2E1"

if __name__ == "__main__":
    rois = Roi.load_from_file(os.path.join(ANNOTATION_PATH, f"{IMAGE_NAME}.geojson"))
    print(rois)

    poly = rois[0].get_polygon_for_level(PyramidalLevel.FIVE)
    poly1 = rois[0].get_polygon_for_level(PyramidalLevel.ZERO)
    b = poly.bounds
    b1 = poly1.bounds
    print(b1)

    open_slide = openslide.OpenSlide(os.path.join(IMAGE_PATH, f"{IMAGE_NAME}.ndpi"))
    print(open_slide.properties)
    image = open_slide.read_region(
        location=(int(b1[0]), int(b1[1])),
        level=PyramidalLevel.FIVE,
        size=(int(b[2] - b[0]), int(b[3] - b[1])),
    )

    plot_rgb(np.array(image), transform_to_bgr=False)
