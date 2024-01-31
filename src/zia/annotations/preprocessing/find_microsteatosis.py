import cv2
import numpy as np

from zia.annotations.config.project_config import ResourcePaths
from zia.annotations.data_sets.upload_predictions import load_polygons
from zia.annotations.preprocessing.polygon_classification import get_foreground_mask
from zia.annotations.preprocessing.stain_normalization import separate_stains, normalizeStaining
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic

if __name__ == "__main__":
    image_name = "FLR-171_0_steatosis_1"
    resourcePaths = ResourcePaths("sample_data")
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.imread(str(resourcePaths.image_path / f"{image_name}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, e = separate_stains(image)

    h = 255 - h
    e = 255 - e
    plot_pic(h)
    plot_pic(e)

    mask = get_foreground_mask(image, low=130)

    dilation = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    plot_pic(dilation)
    _, polygons = load_polygons(resourcePaths.polygon_path / f"{image_name}.geojson")

    contours = [np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2)) for poly in polygons]

    template = np.zeros_like(mask)

    template = cv2.drawContours(template, contours, -1, 255, cv2.FILLED)  # Adjust color and thickness as needed
    template = cv2.dilate(template, kernel, iterations=1)
    template = 255 - template

    result = cv2.bitwise_and(dilation, template)

    plot_pic(result)
