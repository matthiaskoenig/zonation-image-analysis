"""Processing the resulting spatial datasets.

1. Visualization in Napari (i.e. add image layer to napari)
2. Calculation of superpixels and visualization
"""
import napari

from pathlib import Path
from typing import Dict

import numpy as np

from zia.console import console

# 1. read the high dimensional dataset
level = 3
stains = ["HE", "GS", "CYP2E1", "CYP1A2", "CYP3A4", "CYP2D6"]
data_dir_registered: Path = Path(
    f"/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual_registered_L{level}")

results_dir = data_dir_registered / "__results__"
results_dir.mkdir(exist_ok=True, parents=True)

subject_id = "human_UKJ-19-010"


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

# 2. visualize channels in napari with different colors
def view_protein_data(data: Dict[str, np.ndarray]) -> None:
    """View CZI image in napari.

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

    selected_channels = []
    for channel in data.keys():
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

view_protein_data(channel_data)

# 3. calculate superpixels


# 4. try t-sne/umap on the dataset



