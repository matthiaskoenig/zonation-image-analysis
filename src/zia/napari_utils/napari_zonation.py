"""Test reading from zarr store.
"""
import napari

from pathlib import Path
from typing import Dict, List

import numpy as np

from zia.console import console

# ValueError: codec not available: 'imagecodecs_jpeg2k'
from imagecodecs.numcodecs import Jpeg2k
import numcodecs

from zia.data_utils import load_subject_roi_ndarray

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
        name=channel_names,
        rgb=False,
        opacity=1.0,
        # colormap="gray",
        blending="additive",  # translucent
        visible=True
        # multiscale=True,
    )
    napari.run()


if __name__ == "__main__":
    # read the high dimensional dataset
    # subject_id = "UKJ-19-010_Human"
    # subject_id = "MNT-021"
    # subject_id = "SSES2021 9"
    subject_id = "NOR-024"
    roi = 0
    level = 4
    stain_separated_dir: Path = Path(
        # "/media/mkoenig/Extreme Pro/image_data/stain_separated/"
        "/home/mkoenig/data/qualiperf/P3-MetFun/lobulus_segmentation/stain_separated/"
    )

    results = load_subject_roi_ndarray(
        data_dir=stain_separated_dir,
        subject_id=subject_id,
        roi=roi,
        level=level,
    )

    # Visualize
    image: np.ndarray = results["data"]
    channels: List[str] = results["channels"]
    visualize_protein_data(image=image, channel_names=channels)


