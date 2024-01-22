import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
from matplotlib.collections import PatchCollection
from shapely import Polygon
from matplotlib.patches import Polygon as MplPoly

result_dir = Path(r"C:\Users\jonas\Development\git\zonation-image-analysis\data_set")
mask_dir = result_dir / "background_mask"
image_dir = result_dir / "he_image"

image = cv2.imread(str(image_dir / "FLR-168_0_sample_0.png"))
mask = cv2.imread(str(mask_dir / "FLR-168_0_sample_0.png"), cv2.IMREAD_GRAYSCALE)

## get the contours from the opened binary mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# plot_pic(sub_array)
polygons = [Polygon(np.squeeze(cnt)) for cnt in contours if len(cnt) >= 3]

# Example image dimensions
image_width = 10
image_height = 10

# matplotlib.use("QtAgg")
plt.ion()
# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(image_width, image_height))
ax: plt.Axes
ax.imshow(image)

mpl_polygons = [MplPoly(np.vstack(poly.exterior.xy).T) for poly in polygons]
collection = PatchCollection(mpl_polygons, linestyle="-", edgecolor=["red" for i in range(len(polygons))], facecolor="None")
ax.add_collection(collection)

for i in range(len(polygons)):
    collection.get_edgecolors()[i] = (255, 255, 255, 1)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
