import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
    deconvolve_image, find_max_c, create_single_channel_pixels
from zia.data_store import DataStore, ZarrGroups
from zia.io.wsi_tifffile import read_ndpi
from zia.io.zarr_utils import write_to_pyramid, create_pyramid, write_slice_to_zarr_location
from zia.log import get_logger
from threadpoolctl import threadpool_limits

logger = get_logger(__name__)


def get_toplevel_array(image_path: Path) -> zarr.Array:
    arrays = read_ndpi(image_path)
    full_image_array: zarr.Array = arrays[PyramidalLevel.ZERO]

    return full_image_array


def separate_stains(data_store: DataStore, p=0.01, level=PyramidalLevel.ZERO, tile_size: int = 2 ** 12) -> None:
    for roi_no, roi in enumerate(data_store.rois):
        logger.info(f"Start stain separation for ROI {roi_no}")

        roi_cs, roi_rs = roi.get_bound(level)

        roi_h, roi_w = roi_rs.stop - roi_rs.start, roi_cs.stop - roi_cs.start

        # initialize zarr stores to store separated images
        for zg in [ZarrGroups.STAIN_0, ZarrGroups.STAIN_1]:
            data_store.create_pyramid_group(zg, roi_no, (roi_h, roi_w), tile_size, np.uint8)

        # initialize slices
        slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=tile_size)
        final_slices = get_final_slices((roi_rs, roi_cs), slices)

        # this is overhead here since the zarr store itself cannot be pickled.
        zarr_store_address = data_store.image_info.zarr_path
        mask_address = f"{ZarrGroups.LIVER_MASK.name}/{roi_no}/{level.value}"

        t_s = time.time()
        with TemporaryDirectory() as temp_dir, ThreadPoolExecutor(multiprocessing.cpu_count() - 1) as pool:
            with threadpool_limits(limits=1, user_api="blas"):  # TODO: reffactor userapi and make configurable
                samples = pool.map(partial(get_decode_and_save_tile,
                                           image_path=data_store.image_info.path,
                                           temp_dir=temp_dir,
                                           p=p),
                                   enumerate(final_slices),
                                   chunksize=10)

            stack = np.hstack(list(samples)).astype(np.uint8)

            th, _ = cv2.threshold(stack, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            t_e = time.time()

            logger.info(f"Calculate Otsu threshold {th} in {(t_e - t_s) / 60} min.")

            # calculate stain matrix and c max
            t_s = time.time()
            with threadpool_limits(limits=1, user_api="blas"):
                pxi_samples_iterator = pool.map(partial(draw_px_oi_sample,
                                                        temp_dir=temp_dir,
                                                        zarr_store_address=zarr_store_address,
                                                        mask_address=mask_address,
                                                        th=th,
                                                        p=p),
                                                enumerate(slices))

            stack = np.vstack(list(pxi_samples_iterator))

            stain_matrix = calculate_stain_matrix(stack)
            max_c = find_max_c(stack, stain_matrix)

            t_e = time.time()

            print(f"Calculated stain matrix {stain_matrix} and max concentrations {max_c} in {(t_e - t_s) / 60} min.")

            # deconvolute image and save
            t_s = time.time()
            with threadpool_limits(limits=1, user_api="blas"):
                futures = pool.map(partial(deconvolve_and_save,
                                           temp_dir=temp_dir,
                                           stain_matrix=stain_matrix,
                                           max_c=max_c,
                                           zarr_store_address=zarr_store_address,
                                           roi_no=roi_no),
                                   enumerate(slices))

            # access results to get possible exceptions
            for _ in futures:
                pass

        t_e = time.time()

        logger.info(f"Deconvoluted ROI image in {(t_e - t_s) / 60} min")


def deconvolve_and_save(enumerated_tile_slices: Tuple[int, Tuple[slice, slice]],
                        temp_dir: TemporaryDirectory,
                        stain_matrix: np.ndarray,
                        zarr_store_address: str,
                        roi_no: int,
                        max_c: np.ndarray) -> bool:
    i, tile_slices = enumerated_tile_slices
    # read image
    image_tile = load_tile_file(f"{i}", temp_dir)
    # read the index of interesting pixels
    idx = load_tile_file(f"{i}_idx", temp_dir)

    px_oi = image_tile[idx]

    conc = deconvolve_image(px_oi, stain_matrix, max_c)

    for k, c in enumerate(np.vsplit(conc, 2)):
        stain = create_single_channel_pixels(c.reshape(-1))
        template = np.ones(shape=image_tile.shape[:2]).astype(np.uint8) * 255
        template[idx] = stain
        write_slice_to_zarr_location(slice_image=template,
                                     tile_slices=tile_slices,
                                     zarr_store_address=zarr_store_address,
                                     zarr_group=f"stain_{k}",
                                     roi_no=roi_no)

    return True


def draw_px_oi_sample(tile_slices: Tuple[Tuple[slice, slice], str],
                      temp_dir: TemporaryDirectory,
                      zarr_store_address: str,
                      mask_address: str,
                      th: int,
                      p) -> np.ndarray:
    # print("executed")
    i, (rs, cs) = tile_slices

    mask = zarr.convenience.open_array(store=zarr_store_address, path=mask_address)
    tile_mask = mask[rs, cs]

    image_tile = load_tile_file(f"{i}", temp_dir)

    idx = tile_mask & (np.dot(image_tile, np.array([0.587, 0.114, 0.299])) < th)

    temp_file = f"{temp_dir}/{i}_idx.npy"
    np.save(temp_file, idx, allow_pickle=True)

    px_in_mask = image_tile[idx]

    choice = np.random.choice(a=[True, False], size=len(px_in_mask), p=[p, 1 - p])

    px_sample = px_in_mask[choice]

    return px_sample


def get_decode_and_save_tile(final_slices: Tuple[int, Tuple[slice, slice]],
                             image_path: Path,
                             temp_dir: TemporaryDirectory,
                             p: int) -> np.ndarray:
    i, (rs, cs) = final_slices
    image_array = get_toplevel_array(image_path)
    image_tile = image_array[rs, cs]
    file_name = f"{i}"
    np.save(f"{temp_dir}/{file_name}.npy", image_tile, allow_pickle=True)

    choice = np.random.choice(a=[True, False],
                              size=image_tile.shape[:2],
                              p=[p, 1 - p])

    sample = image_tile[choice]
    sample_gs = sample.dot(np.array([0.587, 0.114, 0.299]))
    return sample_gs


def load_tile_file(temp_file: str, temp_dir: TemporaryDirectory) -> np.ndarray:
    return np.load(f"{temp_dir}/{temp_file}.npy", allow_pickle=True)


def calculate_samples_for_otsu(image_tile: np.ndarray, p):
    choice = np.random.choice(a=[True, False],
                              size=image_tile.shape[:2],
                              p=[p, 1 - p])

    sample = image_tile[choice]
    sample_gs = sample.dot(np.array([0.587, 0.114, 0.299]))
    return sample_gs


def get_final_slices(roi_slice: Tuple[slice, slice],
                     tile_slices: List[Tuple[slice, slice]]) -> List[Tuple[slice, slice]]:
    return [get_final_slice(roi_slice, tile_slice) for tile_slice in tile_slices]


def get_final_slice(roi_slice: Tuple[slice, slice], tile_slice: Tuple[slice, slice]) -> \
        Tuple[slice, slice]:
    roi_rs, roi_cs = roi_slice
    rs, cs = tile_slice

    final_rs = slice(roi_rs.start + rs.start, roi_rs.start + rs.stop)
    final_cs = slice(roi_cs.start + cs.start, roi_cs.start + cs.stop)

    return final_rs, final_cs
