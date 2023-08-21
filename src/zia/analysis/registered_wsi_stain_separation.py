import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Dict, Type

import cv2
import numpy as np
import zarr
from threadpoolctl import threadpool_limits

from zia.annotations.annotation.slicing import get_tile_slices, get_final_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.stain_separation.macenko import \
    calculate_stain_matrix, \
    deconvolve_image, find_max_c, create_single_channel_pixels
from zia.data_store import DataStore, ZarrGroups
from zia.io.wsi_tifffile import read_ndpi
from zia.io.zarr_utils import write_slice_to_zarr_location
from zia.log import get_logger

logger = get_logger(__name__)


def get_toplevel_array(image_path: Path) -> zarr.Array:
    arrays = read_ndpi(image_path)
    full_image_array: zarr.Array = arrays[PyramidalLevel.ZERO]

    return full_image_array


def roll_back(data_store: DataStore) -> None:
    """
    function to clean up the directory to prevent invalid data
    @param data_store:
    @return:
    """
    if ZarrGroups.STAIN_0.value in data_store.data.keys():
        del data_store.data[ZarrGroups.STAIN_0.value]

    if ZarrGroups.STAIN_1.value in data_store.data.keys():
        del data_store.data[ZarrGroups.STAIN_1.value]


def separate_stains(path: Path, out_path: Path, p=0.01, level=PyramidalLevel.ZERO,
                    tile_size: int = 2 ** 12) -> None:
    try:
        roi_no = path.parent.name
        subject = path.parent.parent.name
        out = out_path / subject
        zarr_store = zarr.DirectoryStore(str(out))

        logger.info(f"Start stain separation for ROI {roi_no}")

        registered_image = read_ndpi(path)[0]
        roi_h, roi_w = registered_image.shape[:2]
        # initialize slices
        slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=tile_size,
                                 col_first=False)
        # initialize zarr stores to store separated images
        image_pyramids = tuple(
            create_pyramid_group(zarr_store, zg, roi_no, (roi_h, roi_w), tile_size,
                                 np.uint8) for zg in
            [ZarrGroups.STAIN_0, ZarrGroups.STAIN_1])

        # this is overhead here since the zarr store itself cannot be pickled.
        # zarr_store_address = data_store.image_info.zarr_path
        # mask_address = f"{ZarrGroups.LIVER_MASK.name}/{roi_no}/{level.value}"

        t_s = time.time()
        with TemporaryDirectory() as temp_dir, ThreadPoolExecutor(
            multiprocessing.cpu_count() - 1) as pool:
            with threadpool_limits(limits=1,
                                   user_api="blas"):
                samples = pool.map(partial(get_decode_and_save_tile,
                                           image_path=path,
                                           temp_dir=temp_dir,
                                           p=p),
                                   enumerate(slices),
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
                                                        # zarr_store_address=zarr_store_address,
                                                        # mask_address=mask_address,
                                                        th=th,
                                                        p=p),
                                                enumerate(slices))

            stack = np.vstack(list(pxi_samples_iterator))

            stain_matrix = calculate_stain_matrix(stack)
            max_c = find_max_c(stack, stain_matrix)

            t_e = time.time()

            print(
                f"Calculated stain matrix {stain_matrix} and max concentrations {max_c} in {(t_e - t_s) / 60} min.")

            # deconvolute image and save
            t_s = time.time()
            with threadpool_limits(limits=1, user_api="blas"):
                futures = pool.map(partial(deconvolve_and_save,
                                           image_pyramid=image_pyramids,
                                           temp_dir=temp_dir,
                                           stain_matrix=stain_matrix,
                                           max_c=max_c,
                                           zarr_store_address=zarr_store.path
                                           ),
                                   enumerate(slices))

            # access results to get possible exceptions
            for _ in futures:
                pass

        t_e = time.time()
        logger.info(f"Deconvoluted ROI image in {(t_e - t_s) / 60} min")

    except BaseException as e:
        roll_back(data_store)
        raise e


def deconvolve_and_save(enumerated_tile_slices: Tuple[int, Tuple[slice, slice]],
                        image_pyramids: Tuple[Dict[int, str], Dict[int, str]],
                        temp_dir: TemporaryDirectory,
                        stain_matrix: np.ndarray,
                        zarr_store_address: str,
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
                                     image_pyramid=image_pyramids[k],
                                     tile_slices=tile_slices,
                                     zarr_store_address=zarr_store_address)

    return True


def draw_px_oi_sample(tile_slices: Tuple[Tuple[slice, slice], str],
                      temp_dir: TemporaryDirectory,
                      th: int,
                      p,
                      zarr_store_address: str = None,
                      mask_address: str = None,
                      ) -> np.ndarray:
    # print("executed")
    i, (rs, cs) = tile_slices

    # mask = zarr.convenience.open_array(store=zarr_store_address, path=mask_address)
    # tile_mask = mask[rs, cs]

    image_tile = load_tile_file(f"{i}", temp_dir)

    idx = (np.dot(image_tile, np.array([0.587, 0.114, 0.299])) < th)  # & tile_mask

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


def create_pyramid_group(store: zarr.DirectoryStore,
                         zarr_group: ZarrGroups,
                         roi_no: str, shape: Tuple,
                         chunksize: int,
                         dtype: Type) -> Dict[int, str]:
    """Creates a zarr group for a roi to save an image pyramid
    @param zarr_group: the enum to sepcify the zarr subdirectory
    @param roi_no: the number of the ROI
    @param shape: the shape of the array to create
    @param chunksize: the size of the level zero chunk
    @param dtype: the data type stored in the arrays

    """
    root = zarr.group(store, overwrite=True)
    data_group = root.require_group(zarr_group.value)
    roi_group = data_group.require_group(roi_no)
    pyramid_dict: Dict[int, str] = {}

    h, w = shape[:2]

    for i in range(8):
        chunk_w, chunk_h = chunksize, chunksize  # taking the slice size aligns chunks, so that multiprocessing only alway acesses one chunk
        factor = 2 ** i

        new_h, new_w = int(h / factor), int(w / factor)
        new_chunk_h, new_chunk_w = int(chunk_h / factor), int(chunk_w / factor)
        arr: zarr.Array = roi_group.empty(
            str(i),
            shape=(new_h, new_w) + ((shape[2],) if len(shape) == 3 else ()),
            chunks=(new_chunk_w, new_chunk_h) + (
                (shape[2],) if len(shape) == 3 else ()),
            dtype=dtype,
            overwrite=True,
            synchronizer=zarr.ProcessSynchronizer(".chunklock"))

        pyramid_dict[i] = arr.path

    return pyramid_dict
