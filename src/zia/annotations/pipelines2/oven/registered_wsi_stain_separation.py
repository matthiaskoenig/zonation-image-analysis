import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Dict, Type
from imagecodecs.numcodecs import Jpeg2k
import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from threadpoolctl import threadpool_limits

from zia.pipeline.annotation import get_tile_slices
from zia.pipeline.annotation import PyramidalLevel
from zia.pipeline.pipeline_components.algorithm.stain_separation.macenko import \
    calculate_stain_matrix, \
    deconvolve_image, find_max_c, create_single_channel_pixels
from zia.data_store import ZarrGroups
from zia.io.wsi_tifffile import read_ndpi
from zia.io.zarr_utils import write_slice_to_zarr_location
from zia.log import get_logger
import numcodecs

logger = get_logger(__name__)
numcodecs.register_codec(Jpeg2k)


def get_toplevel_array(image_path: Path) -> zarr.Array:
    arrays = read_ndpi(image_path)
    full_image_array: zarr.Array = arrays[PyramidalLevel.ZERO]

    return full_image_array


def roll_back(zarr_store: zarr.DirectoryStore) -> None:
    """
    function to clean up the directory to prevent invalid data
    @param data_store:
    @return:
    """

    logger.info(f"Rollback zarrstore: {zarr_store.path}")

    del zarr_store[ZarrGroups.STAIN_0.value]
    del zarr_store[ZarrGroups.STAIN_1.value]


def separate_stains(path: Path,
                    subject: str,
                    roi_no: str,
                    protein: str,
                    out_path: Path,
                    p=0.01,
                    tile_size: int = 2 ** 12,
                    overwrite=False) -> None:
    out = out_path / f"{subject}.zarr"
    zarr_store = zarr.DirectoryStore(str(out))
    if Path(f"{zarr_store.path}/{ZarrGroups.STAIN_0.value}/{roi_no}/{protein}").exists() and not overwrite:
        print("separated images already exist.")
        return

    logger.info(
        f"Start stain separation for Subject {subject}, ROI {roi_no}, protein: {protein}")

    registered_pyramid = read_ndpi(path)
    for image in registered_pyramid:
        print(np.log2(registered_pyramid[0].shape[0] / image.shape[0]))

    registered_image = read_ndpi(path)[0]
    roi_h, roi_w = registered_image.shape[:2]
    # initialize slices
    slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=tile_size,
                             col_first=False)
    # initialize zarr stores to store separated images
    image_pyramids = tuple(
        create_pyramid_group(zarr_store, zg, roi_no, protein, (roi_h, roi_w), tile_size,
                             np.uint8) for zg in
        [ZarrGroups.STAIN_0, ZarrGroups.STAIN_1])

    t_s = time.time()
    try:
        with TemporaryDirectory() as temp_dir, ThreadPoolExecutor(
                multiprocessing.cpu_count() - 1) as pool:
            with threadpool_limits(limits=1, user_api="blas"):
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

            print(f"Calculated stain matrix {stain_matrix} and max concentrations {max_c} in {(t_e - t_s) / 60} min.")

            # deconvolute image and save
            t_s = time.time()

            ## create a generator of the deconvolved tiles
            deconvolved_tiles = (
                deconvolve(i, temp_dir=temp_dir, stain_matrix=stain_matrix, max_c=max_c)
                for i in range(len(slices)))

            futures = pool.map(partial(save,
                                       image_pyramids=image_pyramids,
                                       zarr_store_address=zarr_store.path,
                                       synchronizer=zarr.sync.ThreadSynchronizer()
                                       ),
                               zip(deconvolved_tiles, slices))

            # access results to get possible exceptions
            for _ in futures:
                pass

        t_e = time.time()
        logger.info(f"Deconvoluted ROI image in {(t_e - t_s) / 60} min")
    except BaseException as e:
        roll_back(zarr_store)
        raise e


def deconvolve(i: int,
               temp_dir: TemporaryDirectory,
               stain_matrix: np.ndarray,
               max_c: np.ndarray) -> list[np.ndarray]:
    # read image
    image_tile = load_tile_file(f"{i}", temp_dir)
    # read the index of interesting pixels
    idx = load_tile_file(f"{i}_idx", temp_dir)

    px_oi = image_tile[idx]

    conc = deconvolve_image(px_oi, stain_matrix, max_c)

    deconvolved = []

    for k, c in enumerate(np.vsplit(conc, 2)):
        stain = create_single_channel_pixels(c.reshape(-1))
        template = np.ones(shape=image_tile.shape[:2]).astype(np.uint8) * 255
        template[idx] = stain
        deconvolved.append(template)

    return deconvolved


def save(deconvolved_slices: Tuple[list[np.ndarray], Tuple[slice, slice]],
         image_pyramids: Tuple[Dict[int, str], Dict[int, str]],
         zarr_store_address: str,
         synchronizer: zarr.sync.ThreadSynchronizer):
    deconvolved, tile_slices = deconvolved_slices
    for k, image in enumerate(deconvolved):
        write_slice_to_zarr_location(slice_image=image,
                                     image_pyramid=image_pyramids[k],
                                     tile_slices=tile_slices,
                                     zarr_store_address=zarr_store_address,
                                     synchronizer=synchronizer)


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
                         roi_no: str,
                         protein: str,
                         shape: Tuple,
                         chunksize: int,
                         dtype: Type) -> Dict[int, str]:
    """Creates a zarr group for a roi to save an image pyramid
    @param zarr_group: the enum to sepcify the zarr subdirectory
    @param roi_no: the number of the ROI
    @param shape: the shape of the array to create
    @param chunksize: the size of the level zero chunk
    @param dtype: the data type stored in the arrays

    """
    root = zarr.group(store)
    data_group = root.require_group(zarr_group.value)
    roi_group = data_group.require_group(roi_no, overwrite=True)
    protein_group = roi_group.require_group(protein)
    pyramid_dict: Dict[int, str] = {}

    h, w = shape[:2]

    for i in range(8):
        chunk_w, chunk_h = chunksize, chunksize  # taking the slice size aligns chunks, so that multiprocessing only always acesses one chunk
        factor = 2 ** i

        new_h, new_w = int(np.ceil(h / factor)), int(np.ceil(w / factor))
        new_chunk_h, new_chunk_w = int(np.ceil(chunk_h / factor)), int(np.ceil(chunk_w / factor))

        if new_chunk_h * new_chunk_w < 1e6:
            new_chunk_h, new_chunk_w = chunk_h, chunk_w

        arr: zarr.Array = protein_group.create(
            str(i),
            shape=(new_h, new_w) + ((shape[2],) if len(shape) == 3 else ()),
            chunks=(new_chunk_w, new_chunk_h) + ((shape[2],) if len(shape) == 3 else ()),
            dtype=dtype,
            overwrite=True,
            write_empty_chunks=False,
            fill_value=255,
            compressor=Blosc(cname="zstd", clevel=10)
        )

        pyramid_dict[i] = arr.path

    return pyramid_dict
