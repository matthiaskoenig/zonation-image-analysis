"""Test CZI reading."""
from pathlib import Path
from typing import List

from zia import CZI_EXAMPLE, CZI_IMAGES_AXIOS, CZI_IMAGES_INITIAL
from zia.czi_io import (
    FluorescenceImage,
    Fluorophor,
    read_czi_images,
    read_czi_images_tifffile,
)


def test_read_czi_example() -> None:
    """Testing reading of CZI example."""
    paths: List[Path] = read_czi_images([CZI_EXAMPLE])
    czi_pickle: Path = paths[0]

    fimage = FluorescenceImage.from_file(czi_pickle)

    # TODO: check the assignments of proteins to the fluorophors
    cyp2e1 = fimage.get_channel_data(Fluorophor.ALEXA_FLUOR_488)
    ecad = fimage.get_channel_data(Fluorophor.CY3)
    dapi = fimage.get_channel_data(Fluorophor.DAPI)

    assert cyp2e1 is not None
    assert ecad is not None
    assert dapi is not None


def test_czi_to_ome_tiff() -> None:
    """
    Testing reformat file from czi to OME-TIFF
    """
    read_czi_images_tifffile(CZI_EXAMPLE)
