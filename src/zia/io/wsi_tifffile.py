"""Read NDPI images.

Handles the metadata correctly and better handling of store
https://github.com/AstraZeneca/napari-wsi

"""
from pathlib import Path
from typing import List

import dask.array as da
import zarr
from tifffile import TiffFileError, ZarrStore, tiffcomment, tifffile

from zia.io.utils import check_image_path


def read_ndpi(image_path: Path) -> List[da.Array]:
    """Read image with tifffile library."""
    check_image_path(image_path)

    # read in zarr store
    store = tifffile.imread(image_path, aszarr=True)
    group = zarr.open(store, mode="r")  # zarr.core.Group or Array

    # FIXME: read metadata
    datasets = group.attrs["multiscales"][0]["datasets"]

    # Load dask array from the zarr storage format
    data = [da.from_zarr(store, component=d["path"]) for d in datasets]
    return data


if __name__ == "__main__":
    from zia import BASE_PATH

    example_ndpi = BASE_PATH / "data" / "ndpi" / "LQF1_LM_HE_PVL.ndpi"

    import dask.array as da

    data: List[da.Array] = read_ndpi(example_ndpi)

    from zia.napari_utils import napari_viewer

    napari_viewer.view_ndpi_data(data=data)
