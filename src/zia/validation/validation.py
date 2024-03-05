from pathlib import Path

import cv2
import numpy as np

from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.common.geometry_utils import GeometryDraw
from zia.pipeline.pipeline_components.algorithm.droplet_detection.droplet_detection import detect_droplets_on_tile

if __name__ == "__main__":
    image_path = Path(__file__).parent.parent.parent.parent / "fat_detection_validation" / "HEs_Masks_steatosis"

    img_path = image_path / "imgs" / "img"
    mask_path = image_path / "mask" / "img"

    for p in img_path.iterdir():
        if not p.is_file():
            continue

        name = p.stem.split("_")[0]
        mask_p = mask_path / f"{name}_mask_HE.png"

        print(p, mask_p)

        img = cv2.imread(str(p))
        mask = cv2.imread(str(mask_p))

        plot_pic(img)
        plot_pic(mask)

        detected_polygons = detect_droplets_on_tile(img)

        true_positves = []
        false_positives = []

        for polygon in detected_polygons:
            mask_template = np.zeros_like(mask, dtype=np.uint8)
            GeometryDraw.draw_geometry(mask_template, polygon, offset=(0, 0), color=True)

            in_common = np.count_nonzero(np.bitwise_and(mask, mask_template))
            detected = np.count_nonzero(mask_template)

            ratio = in_common / detected

            if ratio > 0.5:
                true_positves.append(polygon)
            else:
                false_positives.append(polygon)

        print(len(true_positves) / len(detected_polygons))

