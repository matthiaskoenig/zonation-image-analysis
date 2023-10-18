from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator, Iterator

import numpy as np
import zarr
from PIL import ImageDraw, ImageFont
from tifffile import TiffWriter

from zia import BASE_PATH
from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.slicing import get_tile_slices, get_final_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.config import read_config
from zia.data_store import DataStore
from zia.io.wsi_openslide import read_full_image_from_slide
from zia.io.wsi_tifffile import read_ndpi
from zia.path_utils import FileManager, filter_factory
import cv2


def create_contour(roi_no_data_store: Tuple[int, DataStore],
                   level: PyramidalLevel) -> np.ndarray:
    i, data_store = roi_no_data_store
    img = np.array(data_store.read_full_roi(i, level))
    gs = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    _, thresholded = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    threshold_reverted = cv2.bitwise_not(thresholded)

    mask = draw_roi_poly(i, data_store, level)

    threshold_reverted[~mask] = 0

    return threshold_reverted


def draw_roi_poly(i: int, data_store: DataStore, level: PyramidalLevel) -> np.ndarray:
    roi0 = data_store.rois[i]

    cs0, rs0 = roi0.get_bound(PyramidalLevel.SEVEN)

    img0 = np.zeros(shape=(rs0.stop - rs0.start, cs0.stop - cs0.start),
                    dtype=np.uint8)

    cnr0 = np.array(roi0.get_polygon_for_level(PyramidalLevel.SEVEN, offset=(
        cs0.start, rs0.start)).exterior.coords, np.int32)

    cv2.fillPoly(img0, pts=[cnr0], color=255)

    return img0.astype(bool)


def calculate_hu_distance(reference: Tuple[int, DataStore], test: Tuple[int, DataStore],
                          level: PyramidalLevel = PyramidalLevel.SEVEN):
    ref = create_contour(reference, level)
    test = create_contour(test, level)

    # plot_pic(ref)

    d = cv2.matchShapes(ref, test, cv2.CONTOURS_MATCH_I2, 0)
    return d


def get_mapping_from_distances(distances: np.ndarray):
    mappings = {}

    while len(mappings) < distances.shape[0]:
        r, m = np.unravel_index(np.nanargmin(distances), distances.shape)
        mappings[r] = m
        distances[r, :] = np.nan
        distances[:, m] = np.nan

    return mappings


def _draw_rois(
    liver_rois: List[Roi],
    mapping: Optional[Dict[int, int]],
    image_path: Path,
    data_store: DataStore,
) -> None:
    """Draw rois."""
    region = read_full_image_from_slide(data_store.image, 7)
    draw = ImageDraw.Draw(region)
    font = ImageFont.truetype("arial.ttf", 40)
    print(len(liver_rois))
    for i, liver_roi in enumerate(liver_rois):
        polygon = liver_roi.get_polygon_for_level(
            PyramidalLevel.SEVEN
        )
        poly_points = polygon.exterior.coords
        text_coords = polygon.centroid
        draw.polygon(list(poly_points), outline="red", width=3)
        draw.text((text_coords.x, text_coords.y),
                  str(mapping.get(i)) if mapping else str(i), font=font)

    region.save(image_path, "PNG")


class TileGenerator:
    def __init__(self, slices: List[Tuple[slice, slice]],
                 array: zarr.Array,
                 roi_shape: Tuple[int, int, int],
                 tile_size: int):
        self.slices = slices
        self.array = array
        self.tile_size = tile_size
        self.roi_shape = roi_shape

    def get_tiles(self) -> Iterator[np.ndarray]:
        for rs, cs in self.slices:
            yield self.array[rs, cs]


def write_rois_to_ome_tiff(path: Path,
                           level_generators: Dict[PyramidalLevel, TileGenerator]):
    # remove zero level generator from dict
    zero_generator = level_generators.pop(PyramidalLevel.ZERO)
    print("tilesize ", zero_generator.tile_size)

    with TiffWriter(path, bigtiff=True) as tif:
        metadata = {}
        options = dict(
            photometric='rgb',
            compression='jpeg',
            # resolutionunit='CENTIMETER',
            maxworkers=4)

        print("Write zero")
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
            print(f"Write {level}")
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


if __name__ == "__main__":

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=filter_factory(negative=False)
    )

    report_path = file_manager.results_path / "ordered_rois"
    report_path.mkdir(exist_ok=True)

    image_info_by_subject = file_manager.image_info_grouped_by_subject()

    for subject, image_infos in image_info_by_subject.items():

        print(subject)
        protein_data_store_dict: Dict[str, DataStore] = {
            image_info.metadata.protein: DataStore(image_info)
            for image_info in image_infos}

        reference_data_store = protein_data_store_dict["he"]
        protein_mappings = {}
        if len(reference_data_store.rois) > 1:
            for protein, data_store in protein_data_store_dict.items():
                print(protein)
                distances = np.empty(
                    shape=(len(reference_data_store.rois), len(data_store.rois)))
                for i, roi0 in enumerate(reference_data_store.rois):
                    for k, roi1 in enumerate(data_store.rois):
                        distances[i, k] = calculate_hu_distance(
                            (i, reference_data_store), (k, data_store))
                print(distances)
                protein_mappings[protein] = get_mapping_from_distances(distances)

        for protein, data_store in protein_data_store_dict.items():
            print(protein)
            _draw_rois(data_store.rois, protein_mappings.get(protein),
                       report_path / f"{data_store.image_info.metadata.image_id}.png",
                       data_store)

        for protein, data_store in protein_data_store_dict.items():
            arrays = read_ndpi(data_store.image_info.path)
            zero_array = arrays[0]
            tile_size_log = 12

            roi_generator_dict: Dict[int, Dict[PyramidalLevel, TileGenerator]] = {}

            # create generator for each roi and each resolution levels
            for array in arrays:
                level = np.log2(zero_array.shape[0] / array.shape[0])
                pyramidal_level = PyramidalLevel.get_by_numeric_level(int(level))
                tile_size = 2 ** (tile_size_log - pyramidal_level)

                for k, roi in enumerate(data_store.rois):
                    roi: Roi
                    roi_cs, roi_rs = roi.get_bound(pyramidal_level)
                    roi_h, roi_w = roi_rs.stop - roi_rs.start, roi_cs.stop - roi_cs.start

                    # initialize slices
                    slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=tile_size,
                                             col_first=False)
                    final_slices = get_final_slices((roi_rs, roi_cs), slices)

                    if k not in roi_generator_dict:
                        roi_generator_dict[k] = {}
                    roi_generator_dict[k][pyramidal_level] = TileGenerator(
                        slices=final_slices,
                        array=array,
                        roi_shape=(roi_h, roi_w, 3),
                        tile_size=tile_size)
                    print(k, pyramidal_level, tile_size, (roi_h, roi_w),
                          (roi_rs, roi_cs))

            for i, generator_dict in roi_generator_dict.items():
                roi_no = protein_mappings[protein].get(i) if len(
                    protein_mappings) > 0 else i
                directory = file_manager.image_data_path / "rois_wsi" / subject / str(
                    roi_no)
                directory.mkdir(exist_ok=True, parents=True)
                tiff_path = directory / f"{data_store.image_info.metadata.image_id}.ome.tiff"
                write_rois_to_ome_tiff(tiff_path, generator_dict)
