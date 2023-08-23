"""Processing the resulting spatial datasets.

1. Visualization in Napari (i.e. add image layer to napari)
2. Calculation of superpixels and visualization
"""
import cv2
import napari

from pathlib import Path
from typing import Dict, List

import numpy as np
from skimage import img_as_float

from zia.console import console


def visualize_protein_data(image: np.ndarray, channel_names: List[str]) -> None:
    """Visualization of channel layers in napari.

    This is starting napari and blocking.

    Each channel in a multichannel image can be displayed as an individual layer by
    using the channel_axis argument in viewer.add_image(). All the rest of the
    arguments to viewer.add_image() (e.g. name, colormap, contrast_limit) can take
    the form of a list of the same size as the number of channels.

    blending: A translucent setting will cause the layer to blend with the layers below
    it if you decrease its opacity but will fully block those layers if its opacity is
    1. This is a reasonable default, useful for many applications.
    """
    viewer = napari.Viewer()

    console.print(f"Image: {image.dtype}", image.shape)

    viewer.add_image(
        image,
        channel_axis=2,
        name=selected_channels,
        rgb=False,
        opacity=1.0,
        # colormap="gray",
        blending="additive",  # translucent
        visible=True
        # multiscale=True,
    )
    napari.run()

# 1. read the high dimensional dataset
level = 3
stains = ["HE", "GS", "CYP2E1", "CYP1A2", "CYP3A4", "CYP2D6"]
data_dir_registered: Path = Path(
    f"/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual_registered_L{level}")

results_dir = data_dir_registered / "__results__"
results_dir.mkdir(exist_ok=True, parents=True)

subject_id = "human_UKJ-19-010"
# subject_id = "mouse_MNT-021"
# subject_id = "pig_SSES2021-9"


# get channels
channels = []
for stain in stains:
    channel_name_0 = f"{stain}_H"
    if stain == "HE":
        channel_name_1 = f"{stain}_E"
    else:
        channel_name_1 = f"{stain}_DOB"

    for name in [channel_name_0, channel_name_1]:
        channels.append(name)

print(f"Channels: {channels}")

# combine image
channel_data: Dict[str, np.ndarray] = {}
for channel in channels:
    npy_path = results_dir / f"{subject_id}_{channel}.npy"
    console.print(npy_path)
    data = np.load(npy_path)
    console.print(f"{data.dtype}", data.shape)
    channel_data[channel] = data

print(channel_data.keys())

# selected channels
selected_channels = []
for channel in channel_data.keys():
    if channel.endswith("_DOB") or "_HE_" in channel:
        selected_channels.append(channel)

channel_0 = selected_channels[0]
shape = channel_data[channel_0].shape
dtype = channel_data[channel_0].dtype
n_channels = len(selected_channels)
print(f"Channels: {n_channels}")
image = np.zeros(shape=(shape[0], shape[1], n_channels), dtype=dtype)
for k, channel in enumerate(selected_channels):
    image[:, :, k] = 255 - channel_data[channel]


# -----------------------------------------------------------------------------------
# # 3. calculate superpixels
# # https://pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
# console.rule(title="Superpixel calculation", align="left", style="white")
# from skimage.segmentation import slic, mark_boundaries
# from skimage.measure import regionprops
# from matplotlib import pyplot as plt
#
# # Assume image is given
# n_segments = 200
# # segments = slic(image, n_segments=n_segments, compactness=0.1, enforce_connectivity=True)
#
# # load the image and apply SLIC and extract (approximately)
# # the supplied number of segments
# # image = cv2.imread(args["image"])
# segments = slic(img_as_float(image), n_segments=n_segments, sigma=5)
#
#
# # show the output of SLIC
# fig = plt.figure("Superpixels")
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
# plt.axis("off")
# plt.show()
# console.rule(style="white")

# # region properties
# props = regionprops(segments, intensity_image=image)
#
# segmentsToExclude = []
# for s, segment in enumerate(segments):
#     if props[s].mean_intensity < 5:  # basically black
#         segmentsToExclude.append(s)
# -----------------------------------------------------------------------------------


# Visualize
visualize_protein_data(image=image, channel_names=selected_channels)

# -----------------------------------------------------------------------------------
# UMAP
# 4. try t-sne/umap on the dataset
# https://pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
console.rule(title="Umap calculation", align="left", style="white")






