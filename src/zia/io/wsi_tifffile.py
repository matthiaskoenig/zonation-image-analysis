"""Read NDPI images.

Handles the metadata correctly and better handling of store
https://github.com/AstraZeneca/napari-wsi

"""
from pathlib import Path
from typing import List

import zarr
from tifffile import tifffile, TiffPage

from zia.io.utils import check_image_path


def read_ndpi(image_path: Path, chunkshape=(2 ** 11, 2 ** 11, 3)) -> List[zarr.Array]:
    """Read image with tifffile library."""
    check_image_path(image_path)
    print("open zarr store")
    # read in zarr store
    store: tifffile.ZarrTiffStore = tifffile.imread(image_path, aszarr=True)
    print(type(store))
    print(store.keys())
    group = zarr.open(store, mode="r")  # zarr.core.Group or Array
    # FIXME: read metadata
    datasets = group.attrs["multiscales"][0]["datasets"]
    print(group.attrs["multiscales"])

    # Load dask array from the zarr storage format
    data = [group.get(d["path"]) for d in datasets]
    return data


if __name__ == "__main__":
    example_ndpi = Path(
        r"D:\data\cyp_species_comparison\control\human\CYP2E1\UKJ-19-036_Human _J-19-0488_CYP2E1-1 800_Run 15__MAA_006.ndpi")

    data = read_ndpi(example_ndpi)
    for d in data:
        print(d.shape)
