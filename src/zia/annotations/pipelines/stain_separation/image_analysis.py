import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List, Dict

import cv2
import numpy as np
import zarr

from zia.annotations.annotation.slicing import get_tile_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.stain_separation.macenko import \
    calculate_stain_matrix, \
    deconvolve_image, find_max_c
from zia.data_store import DataStore, ZarrGroups
from zia.io.wsi_tifffile import read_ndpi
from zia.io.zarr_utils import write_to_pyramid
from zia.log import get_logger

logger = get_logger(__name__)


def get_toplevel_array(image_path: Path) -> zarr.Array:
    arrays = read_ndpi(image_path)
    full_image_array: zarr.Array = arrays[PyramidalLevel.ZERO]

    return full_image_array


def separate_stains(data_store: DataStore, p=0.01, level=PyramidalLevel.ZERO) -> None:

    arrays = read_ndpi(data_store.image_info.path)
    full_image_array: zarr.Array = arrays[level]

    pool1 = ThreadPoolExecutor(multiprocessing.cpu_count())
    pool2 = ThreadPoolExecutor(multiprocessing.cpu_count())
    pool3 = ThreadPoolExecutor(multiprocessing.cpu_count())
    pool4 = ThreadPoolExecutor(multiprocessing.cpu_count())

    for roi_no, roi in enumerate(data_store.rois):
        logger.info(f"Start stain separation for ROI {roi_no}")

        mask = data_store.get_array(ZarrGroups.LIVER_MASK, roi_no, level)

        roi_cs, roi_rs = roi.get_bound(level)

        roi_h, roi_w = roi_rs.stop - roi_rs.start, roi_cs.stop - roi_cs.start

        zarr_group = ZarrGroups.E_STAIN if data_store.image_info.metadata.protein == "he" else ZarrGroups.DAB_STAIN
        # create pyramidal group to persist mask
        pyramid_dict_dab = data_store.create_pyramid_group(
            zarr_group, roi_no, (roi_h, roi_w), np.uint8
        )

        pyramid_dict_he = data_store.create_pyramid_group(
            ZarrGroups.H_STAIN, roi_no, (roi_h, roi_w), np.uint8
        )

        slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=2 ** 12)
        final_slices = get_final_slices((roi_rs, roi_cs), slices)

        # compute otsu threshold

        t_s = time.time()
        print(len(slices))
        with TemporaryDirectory() as temp_dir:
            tile_images = pool1.map(
                partial(get_decode_and_save_tile, image_array=full_image_array,
                        temp_dir=temp_dir), enumerate(final_slices), chunksize=10)

            samples = pool2.map(
                partial(calculate_samples_for_otsu, p=p),
                tile_images)

            stack = np.hstack(list(samples)).astype(np.uint8)

            th, _ = cv2.threshold(stack, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            t_e = time.time()

            logger.info(f"Calculate Otsu threshold {th} in {(t_e - t_s) / 60} min.")

            # calculate stain matrix and c max
            t_s = time.time()

            pxi_samples_iterator = pool3.map(
                partial(draw_px_oi_sample, temp_dir=temp_dir, mask=mask, th=th, p=p),
                enumerate(slices))

            stack = np.vstack(list(pxi_samples_iterator))

            stain_matrix = calculate_stain_matrix(stack)
            maxC = find_max_c(stack, stain_matrix)

            print(stain_matrix, maxC)

            t_e = time.time()

            print(
                f"Calculated stain matrix and max concentrations in {(t_e - t_s) / 60} min.")

            # deconvolute image and save
            t_s = time.time()

            px_ois = pool4.map(partial(read_pixels_of_interest, temp_dir=temp_dir),
                               range(len(slices)))

            for i, (image_tile, idx) in px_ois:
                px_oi = image_tile[idx]

                template_dab = np.ones(shape=image_tile.shape[:2]).astype(
                    np.uint8) * 255
                template_h = np.ones(shape=image_tile.shape[:2]).astype(np.uint8) * 255

                hematoxylin, dab = deconvolve_image(px_oi, stain_matrix, maxC)
                print(f"deconvolved {i}")

                # print(dab)
                template_dab[idx] = dab
                template_h[idx] = hematoxylin
                write_to_zarr(slices[i], template_h, template_dab,
                              pyramid_dicts=(pyramid_dict_dab, pyramid_dict_he))
                print(f"saved {i}")
            t_e = time.time()

            logger.info(f"Deconvoluted ROI image in {(t_e - t_s) / 60} min")


def read_pixels_of_interest(i: int,
                            temp_dir: TemporaryDirectory):
    # read image
    print(i)
    ts = time.time()
    image_tile = load_tile_file(f"{i}", temp_dir)
    te = time.time()
    print(f"Loaded tile in {te - ts} s.")
    idx = load_tile_file(f"{i}_idx", temp_dir)

    return i, (image_tile, idx)


def write_to_zarr(tile_slices: Tuple[slice, slice], he: np.ndarray, dab: np.ndarray,
                  pyramid_dicts: Tuple[Dict[int, zarr.Array], Dict[int, zarr.Array]]):
    rs, cs = tile_slices

    pyramid_dict_dab, pyramid_dict_he = pyramid_dicts
    write_to_pyramid(dab, pyramid_dict_dab, rs, cs)
    write_to_pyramid(he, pyramid_dict_he, rs, cs)


def draw_px_oi_sample(tile_slices: Tuple[Tuple[slice, slice], str],
                      temp_dir: TemporaryDirectory,
                      mask: zarr.Array,
                      th: int,
                      p) -> np.ndarray:
    # print("executed")
    i, (rs, cs) = tile_slices

    ts = time.time()
    tile_mask = mask[rs, cs]
    te = time.time()

    # read image tile from the open slide
    # print(f"tile mask loaded in {te - ts} s.")

    ts = time.time()
    image_tile = load_tile_file(f"{i}", temp_dir)
    te = time.time()

    # print(f"image tile mask loaded in {te - ts} s.")

    # pixels in mask and otsu filtered

    idx = tile_mask & (np.dot(image_tile, np.array([0.587,
                                                    0.114,
                                                    0.299])) < th)

    temp_file = f"{temp_dir}/{i}_idx.npy"
    np.save(temp_file, idx, allow_pickle=True)

    px_in_mask = image_tile[idx]

    choice = np.random.choice(a=[True, False], size=len(px_in_mask),
                              p=[p, 1 - p])

    px_sample = px_in_mask[choice]

    return px_sample


def get_decode_and_save_tile(final_slices: Tuple[int, Tuple[slice, slice]],
                             image_array: zarr.Array,
                             temp_dir: TemporaryDirectory) -> np.ndarray:
    i, (rs, cs) = final_slices
    ts = time.time()
    image_tile = image_array[rs, cs]
    te = time.time()
    print(f"Loaded tile in {te - ts} s.")
    file_name = f"{i}"
    np.save(f"{temp_dir}/{file_name}.npy", image_tile, allow_pickle=True)
    return image_tile


def load_tile_file(temp_file: str, temp_dir: TemporaryDirectory) -> np.ndarray:
    return np.load(f"{temp_dir}/{temp_file}.npy", allow_pickle=True)


def calculate_samples_for_otsu(image_tile: np.ndarray,
                               p):
    choice = np.random.choice(a=[True, False],
                              size=image_tile.shape[:2],
                              p=[p, 1 - p])

    sample = image_tile[choice]
    sample_gs = sample.dot(np.array([0.587, 0.114, 0.299]))
    return sample_gs


def get_final_slices(roi_slice: Tuple[slice, slice],
                     tile_slices: List[Tuple[slice, slice]]) -> List[
    Tuple[slice, slice]]:
    return [get_final_slice(roi_slice, tile_slice) for tile_slice in tile_slices]


def get_final_slice(roi_slice: Tuple[slice, slice], tile_slice: Tuple[slice, slice]) -> \
    Tuple[slice, slice]:
    roi_rs, roi_cs = roi_slice
    rs, cs = tile_slice

    final_rs = slice(roi_rs.start + rs.start, roi_rs.start + rs.stop)
    final_cs = slice(roi_cs.start + cs.start, roi_cs.start + cs.stop)

    return final_rs, final_cs
