"""Read NDPI images.

Handles the metadata correctly and better handling of store
https://github.com/AstraZeneca/napari-wsi

"""
import os
from pathlib import Path
from typing import List, Dict

import zarr
from tifffile import tifffile, TiffWriter

from zia.log import get_logger
from zia.pipeline.common.resolution_levels import PyramidalLevel
from zia.pipeline.common.tile_generator import TileGenerator
from zia.io.utils import check_image_path

log = get_logger(__file__)


def read_ndpi(image_path: Path, max_workers=os.cpu_count() - 1) -> List[zarr.Array]:
    """Read image with tifffile library."""
    check_image_path(image_path)
    # read in zarr store
    store: tifffile.ZarrTiffStore = tifffile.imread(str(image_path), aszarr=True, maxworkers=max_workers)
    group = zarr.open(store, mode="r")  # zarr.core.Group or Array
    # FIXME: read metadata
    datasets = group.attrs["multiscales"][0]["datasets"]
    # Load dask array from the zarr storage format
    data = [group.get(d["path"]) for d in datasets]
    return data


def write_rois_to_ome_tiff(path: Path,
                           level_generators: Dict[PyramidalLevel, TileGenerator]):
    # remove zero level generator from dict
    zero_generator = level_generators.pop(PyramidalLevel.ZERO)
    log.info(f"Writing {path.name} with tile size {zero_generator.tile_size}")

    with TiffWriter(path, bigtiff=True) as tif:
        metadata = {}
        options = dict(
            photometric='rgb',
            compression='jpeg',
            # resolutionunit='CENTIMETER',
            maxworkers=os.cpu_count())

        log.info(f"Write level {PyramidalLevel.ZERO}")
        # writes the level zero resolution
        tif.write(
            data=zero_generator.get_tiles(),
            shape=zero_generator.roi_shape,
            dtype=zero_generator.array.dtype,
            tile=(zero_generator.tile_size, zero_generator.tile_size),
            subifds=len(level_generators),

            # resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            # metadata=metadata,
            **options
        )

        # write pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
        for level, tile_generator in level_generators.items():
            log.info(f"Write level {level}")
            tif.write(
                data=tile_generator.get_tiles(),
                shape=tile_generator.roi_shape,
                dtype=tile_generator.array.dtype,
                tile=(tile_generator.tile_size, tile_generator.tile_size),
                subfiletype=1,
                **options
            )
        # resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize), **options)
        # add a thumbnail image as a separate series
        # it is recognized by QuPath as an associated image

        # thumbnail = (data[0, 0, ::8, ::8] >> 2).astype('uint8')

        # tif.write(thumbnail, metadata={'Name': 'thumbnail'})
