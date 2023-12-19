import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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

from zia.annotations.annotation.slicing import get_tile_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.stain_separation.macenko import \
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


def transfer_and_choose_sample(final_slices: Tuple[int, Tuple[slice, slice]],
                               source_array: zarr.Array,
                               target_array: zarr.Array,
                               p: float
                               ) -> np.ndarray:
    rs, cs = final_slices
    chunk = source_array[rs, cs]
    target_array[rs, cs] = source_array[rs, cs]
    return choose_pixels_from_tiles(chunk, p)


def choose_pixels_from_tiles(chunk: np.ndarray,
                             p: float) -> np.ndarray:
    n = np.prod(chunk.shape[:2])
    random_indices = np.random.choice(n, size=int(n * p), replace=False)

    row_idc, col_idc = np.unravel_index(random_indices, chunk.shape[:2])

    sampled_pixels = chunk[row_idc, col_idc]
    return sampled_pixels


def choose_fg_pixels(final_slices: Tuple[slice, slice],
                     mmep_array: zarr.Array,
                     mask_array: zarr.Array,
                     th: int,
                     p: float) -> np.ndarray:
    rs, cs = final_slices
    chunk = mmep_array[rs, cs]
    fg_mask = chunk.dot([0.587, 0.114, 0.299]) < th
    mask_array[rs, cs] = fg_mask
    chunk_fg = chunk[fg_mask]
    choice = np.random.choice(len(chunk_fg), size=int(len(chunk_fg) * p), replace=False)
    return chunk_fg[choice]


def deconvolve(final_slices: Tuple[slice, slice],
               stain_matrix: np.ndarray,
               max_c: np.ndarray,
               mmep_array: zarr.Array,
               mask_array: zarr.Array,
               ) -> list[np.ndarray]:
    rs, cs = final_slices
    image_tile = mmep_array[rs, cs]
    # read the index of interesting pixels
    idx = mask_array[rs, cs]

    px_oi = image_tile[idx]

    conc = deconvolve_image(px_oi, stain_matrix, max_c)

    deconvolved = []

    for k, c in enumerate(np.vsplit(conc, 2)):
        stain = create_single_channel_pixels(c.reshape(-1))
        template = np.ones(shape=image_tile.shape[:2]).astype(np.uint8) * 255
        template[idx] = stain
        deconvolved.append(template)

    return deconvolved


def separate_stains(path: Path,
                    subject: str,
                    roi_no: str,
                    protein: str,
                    out_path: Path,
                    executor: ThreadPoolExecutor,
                    p=0.001,
                    overwrite=False) -> None:
    with zarr.storage.TempStore() as temp_store:

        out = out_path / f"{subject}.zarr"
        zarr_store = zarr.DirectoryStore(str(out))

        if Path(f"{zarr_store.path}/{ZarrGroups.STAIN_0.value}/{roi_no}/{protein}").exists() and not overwrite:
            print("separated images already exist.")
            return

        try:
            run_stain_separation(path,
                                 zarr_store,
                                 subject,
                                 roi_no,
                                 protein,
                                 executor,
                                 temp_store,
                                 p)
        except BaseException as e:
            roll_back(zarr_store)
            raise e


def run_stain_separation(path: Path,
                         zarr_store: zarr.DirectoryStore,
                         subject: str,
                         roi_no: str,
                         protein: str,
                         executor: ThreadPoolExecutor,
                         temp_store: zarr.TempStore,
                         p: float) -> None:
    logger.info(
        f"Start stain separation for Subject {subject}, ROI {roi_no}, protein: {protein}")

    registered_pyramid = read_ndpi(path)

    t_s = time.time()

    registered_image = read_ndpi(path)[0]

    tile_size = registered_image.chunks[:2]

    roi_h, roi_w = registered_image.shape[:2]

    # initialize zarr stores to store separated images
    image_pyramids = tuple(create_pyramid_group(zarr_store,
                                                zg,
                                                roi_no,
                                                protein,
                                                (roi_h, roi_w),
                                                tile_size,
                                                np.uint8) for zg in
                           [ZarrGroups.STAIN_0, ZarrGroups.STAIN_1])

    # initialize slices
    slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=tile_size,
                             col_first=False)

    # create uncompressed memory mapped array
    mem_mapped_array = zarr.open(store=temp_store,
                                 mode="w",
                                 path="uncrompressed",
                                 shape=registered_image.shape,
                                 dtype=registered_image.dtype,
                                 compressor=None,
                                 chunks=registered_image.chunks)

    mem_mapped_mask = zarr.open(store=temp_store,
                                mode="w",
                                path="mask",
                                compressor=None,
                                shape=registered_image.shape[:2],
                                dtype=bool,
                                chunks=registered_image.chunks[:2])

    # assign image array uncompressed to memory map

    result_iterator = executor.map(partial(transfer_and_choose_sample,
                                           source_array=registered_image,
                                           target_array=mem_mapped_array,
                                           p=p),
                                   slices)

    pixel_sample = np.vstack(list(result_iterator)).astype(np.uint8)
    th, _ = cv2.threshold(pixel_sample, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    t_e = time.time()
    t_final = (t_e - t_s) / 60

    logger.info(f"Decompressed image and calulated threshold {th} in {t_final:.2f} min.")

    t_s = time.time()

    result_iterator = executor.map(partial(choose_fg_pixels,
                                           mmep_array=mem_mapped_array,
                                           mask_array=mem_mapped_mask,
                                           th=th,
                                           p=p),
                                   slices)

    pixel_sample = np.vstack(list(result_iterator)).astype(np.uint8)
    stain_matrix = calculate_stain_matrix(pixel_sample)
    max_c = find_max_c(pixel_sample, stain_matrix)

    t_e = time.time()
    t_final = (t_e - t_s) / 60
    logger.info(f"Calculated stain matrix {stain_matrix} and max concentrations {max_c} in {t_final:.2f} min.")

    t_s = time.time()
    ## create a generator of the deconvolved tiles
    deconvolved_tiles = (deconvolve(slice_tuple,
                                    stain_matrix=stain_matrix,
                                    max_c=max_c,
                                    mmep_array=mem_mapped_array,
                                    mask_array=mem_mapped_mask)
                         for slice_tuple in slices)

    futures = executor.map(partial(save,
                                   image_pyramids=image_pyramids,
                                   zarr_store_address=zarr_store.path,
                                   synchronizer=zarr.sync.ThreadSynchronizer()
                                   ),
                           zip(deconvolved_tiles, slices))

    # access results to get possible exceptions
    for _ in futures:
        pass

    t_e = time.time()
    t_final = (t_e - t_s) / 60
    logger.info(f"Deconvoluted and saved image in {t_final:.2f} min.")


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


def create_pyramid_group(store: zarr.DirectoryStore,
                         zarr_group: ZarrGroups,
                         roi_no: str,
                         protein: str,
                         shape: Tuple,
                         chunksize: Tuple[int, int],
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
        chunk_h, chunk_w = chunksize  # taking the slice size aligns chunks, so that multiprocessing only alway acesses one chunk
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
            compressor=Blosc(cname="zstd", clevel=8)
        )

        pyramid_dict[i] = arr.path

    return pyramid_dict
