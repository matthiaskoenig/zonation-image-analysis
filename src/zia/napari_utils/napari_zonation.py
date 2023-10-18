"""Test reading from zarr store.
"""
import cv2
import napari

from pathlib import Path
from typing import Dict, List

import numpy as np
from skimage import img_as_float
import zarr

from zia.console import console
# ValueError: codec not available: 'imagecodecs_jpeg2k'
from imagecodecs.numcodecs import Jpeg2k
import numcodecs
numcodecs.register_codec(Jpeg2k)

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
# subject_id = "UKJ-19-010_Human"
# subject_id = "MNT-021"
# subject_id = "SSES2021 9"
subject_id = "NOR-024"
roi = 0

data_dir_registered_stain_separated: Path = Path(
    # "/media/mkoenig/Extreme Pro/image_data/stain_separated/"
    "/home/mkoenig/data/qualiperf/P3-MetFun/lobulus_segmentation/stain_separated/"
)
stains = ["HE", "GS", "CYP2E1", "CYP1A2", "CYP3A4", "CYP2D6"]

zarr_store_path = data_dir_registered_stain_separated / f"{subject_id}.zarr"
level = 2


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

console.print(f"Channels: {channels}")


# selected channels
selected_channels = []
for channel in channels:
    if channel.endswith("_DOB"):
        selected_channels.append(channel)

console.print(f"Selected channels: {selected_channels}")

# load data and combine in image

def load_ndarray(store, stain, roi, protein, level) -> np.ndarray:
    """Load ndarray data."""
    array_path = f"stain_{stain}/{roi}/{protein}/{level}"
    data: zarr.Array = zarr.open(store=zarr_store_path, path=array_path)
    console.print(data)
    return np.array(data)


channel_data: Dict[str, np.ndarray] = {}

for channel in channels:
    # load data
    protein = channel.split("_")[0]
    if channel.endswith("_DOB"):
        stain = 1
    elif channel.endswith("_H"):
        stain = 0

    channel_data[channel] = load_ndarray(
        store=zarr_store_path,
        stain=stain,
        roi=roi,
        protein=protein,
        level=level,
    )


channel_0 = selected_channels[0]
shape = channel_data[channel_0].shape
dtype = channel_data[channel_0].dtype
n_channels = len(selected_channels)
print(f"Channels: {n_channels}")
image = np.zeros(shape=(shape[0], shape[1], n_channels), dtype=dtype)
for k, channel in enumerate(selected_channels):
    image[:, :, k] = 255 - channel_data[channel]  # store inverted image



# Visualize
visualize_protein_data(image=image, channel_names=selected_channels)
