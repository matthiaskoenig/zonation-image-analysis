"""Utilities to load and work with the stain separated images.

Example given data loading, caching, pre-processing for spatial PCA analysis.

"""
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from zia.console import console
import zarr


def load_ndarray(store: str, stain, roi, protein, level) -> np.ndarray:
    """Load ndarray data."""
    array_path = f"stain_{stain}/{roi}/{protein}/{level}"
    data: zarr.Array = zarr.open(store=store, path=array_path)
    console.print(data)
    return np.array(data)

def load_subject_roi_ndarray(
    data_dir: Path, subject_id: str, roi: int, level: int, stains: List[str] = None) -> Dict[str, Any]:
    """Load stain separated ndarray."""
    if not stains:
        stains = ["HE", "GS", "CYP2E1", "CYP1A2", "CYP3A4", "CYP2D6"]

    zarr_store_path = data_dir / f"{subject_id}.zarr"

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

    results: Dict[str, Any] = {
        "channels": selected_channels,
        "data": image,
    }
    return results

def get_all_rois():
    """Get all rois from the stain separated data."""
    # Prepare cached data set
    pass

if __name__ == "__main__":
    # read the high dimensional dataset
    # subject_id = "UKJ-19-010_Human"
    # subject_id = "MNT-021"
    # subject_id = "SSES2021 9"
    subject_id = "NOR-024"
    roi = 0
    level = 2
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
    console.print(results["channels"])
    console.print(results["data"].shape)


