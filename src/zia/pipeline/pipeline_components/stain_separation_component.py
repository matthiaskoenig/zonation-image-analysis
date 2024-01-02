import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, Type, List, Set, Optional

import cv2
import numcodecs
import numpy as np
import zarr
from imagecodecs.numcodecs import Jpeg
from imagecodecs.numcodecs import Jpeg2k

from zia.config import Configuration
from zia.io.wsi_tifffile import read_ndpi
from zia.io.zarr_utils import write_slice_to_zarr_location
from zia.log import get_logger
from zia.pipeline.common.resolution_levels import PyramidalLevel
from zia.pipeline.common.slicing import get_tile_slices
from zia.pipeline.file_management.file_management import Slide, SlideFileManager
from zia.pipeline.pipeline_components.algorithm.stain_separation.macenko import \
    calculate_stain_matrix, \
    deconvolve_image, find_max_c, create_single_channel_pixels
from zia.pipeline.pipeline_components.pipeline import IPipelineComponent
from zia.pipeline.pipeline_components.roi_registration_component import SlideRegistrationComponent

logger = get_logger(__name__)
numcodecs.register_codec(Jpeg2k)
numcodecs.register_codec(Jpeg)


def get_registered_slide(lobe_dir: Path, slide: Slide) -> Optional[Path]:
    ome_tiff_file = next(lobe_dir.glob(f"{slide.name}.ome.tiff"), None)
    if ome_tiff_file is None:
        return None

    return ome_tiff_file


class Stain(IntEnum):
    ZERO = 0
    ONE = 1

    @classmethod
    def get_by_numeric_level(cls, stain: int) -> "Stain":
        return Stain(int(stain))


class StainSeparationComponent(IPipelineComponent):
    """Pipeline step for stain separation."""
    dir_name = "StainSeparation"

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, stains: List[Stain], overwrite: bool = False):
        super().__init__(project_config, file_manager, StainSeparationComponent.dir_name, overwrite)
        self.stain_path_dict = self.init_stain_paths(stains)
        logger.info(f"Initialized StainSeparationComponent with stain {stains}.")

    def init_stain_paths(self, stains: List[Stain]):
        stain_path_dict = {}
        for stain in stains:
            p = self.get_image_data_path() / f"{stain.value}"
            p.mkdir(exist_ok=True, parents=True)
            stain_path_dict[stain] = p
        return stain_path_dict

    def get_separation_path_zarr_path(self, slide, lobe_id: str) -> Dict[Stain, Path]:
        paths = {}
        for stain, stain_path in self.stain_path_dict.items():
            p = stain_path / f"{slide.subject}/{lobe_id}/{slide.protein}.zarr"
            p.mkdir(exist_ok=True, parents=True)
            paths[stain] = p
        return paths

    def filter_existing(self, zarr_paths: Dict[Stain, Path], slide: Slide, lobe_id: str) -> Dict[Stain, Path]:
        if self.overwrite:
            return zarr_paths

        filtered = {}
        for stain, zarr_path in zarr_paths.items():
            if any(zarr_path.iterdir()):
                logger.info(f"stain separation for subject: {slide.subject}, protein: {slide.protein}, "
                            f"lobe: {lobe_id} already exists.")
            else:
                filtered[stain] = zarr_path
        return filtered

    def get_roi_dir(self, subject: str) -> Dict[str, Path]:
        p = self.project_config.image_data_path / SlideRegistrationComponent.dir_name / subject
        if not p.exists():
            raise FileNotFoundError(f"No ROI registration directory found for {subject}")

        lobe_dir_dict = {sub_dir.name: sub_dir for sub_dir in p.iterdir()}

        if len(lobe_dir_dict) == 0:
            raise FileExistsError(f"No registered ROI tiff files found for {subject}")

        return lobe_dir_dict

    def run(self) -> None:
        with ThreadPoolExecutor(os.cpu_count()) as executor:
            for subject, slides in self.file_manager.group_by_subject().items():
                for lobe_id, lobe_dir in self.get_roi_dir(subject).items():
                    for slide in slides:

                        registered_slide = get_registered_slide(lobe_dir, slide)

                        if registered_slide is None:
                            logger.warning(f"No registered ROI image found for {slide.subject}, {slide.species}.")
                            continue

                        zarr_dirs = self.get_separation_path_zarr_path(slide, lobe_id)

                        zarr_dirs = self.filter_existing(zarr_dirs, slide, lobe_id)

                        if len(zarr_dirs) == 0:
                            continue

                        logger.info(f"Start stain separation for Subject {subject}, ROI {lobe_id}, protein: {slide.protein}")

                        separate_stains(zarr_dirs, registered_slide, executor)


def separate_stains(zarr_paths: Dict[Stain, Path],
                    registered_slide: Path,
                    executor: ThreadPoolExecutor,
                    p: float = 0.01) -> None:
    zarr_stores = {stain: zarr.DirectoryStore(str(zarr_path)) for stain, zarr_path in zarr_paths.items()}
    temp_store = zarr.TempStore()

    try:
        run_stain_separation(zarr_stores, registered_slide, temp_store, executor, p)
    except BaseException as e:
        roll_back(zarr_stores)
        raise e
    finally:
        logger.info(f"Clearing temporary zarr store.")
        temp_store.clear()
        temp_store.close()  # this is not implemented for this kind of zarr store


def get_toplevel_array(image_path: Path) -> zarr.Array:
    arrays = read_ndpi(image_path)
    full_image_array: zarr.Array = arrays[PyramidalLevel.ZERO]

    return full_image_array


def roll_back(zarr_stores: Dict[Stain, zarr.DirectoryStore]) -> None:
    """
    function to clean up the directory to prevent invalid data
    """

    for zarr_store in zarr_stores.values():
        logger.info(f"Rollback zarr store: {zarr_store.path}")
        zarr_store.clear()


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
               stains: Set[Stain]
               ) -> Dict[Stain, np.ndarray]:
    rs, cs = final_slices
    image_tile = mmep_array[rs, cs]
    # read the index of interesting pixels
    idx = mask_array[rs, cs]

    px_oi = image_tile[idx]

    conc = deconvolve_image(px_oi, stain_matrix, max_c)
    conc_split = np.vsplit(conc, 2)

    deconvolved = {}
    for stain in stains:
        c = conc_split[stain]
        stain_c = create_single_channel_pixels(c.reshape(-1))
        template = np.ones(shape=image_tile.shape[:2]).astype(np.uint8) * 255
        template[idx] = stain_c
        deconvolved[stain] = template

    return deconvolved


def run_stain_separation(zarr_stores: Dict[Stain, zarr.DirectoryStore],
                         registered_image_path: Path,
                         temp_store: zarr.TempStore,
                         executor: ThreadPoolExecutor,
                         p: float) -> None:
    t_s = time.time()

    registered_image = read_ndpi(registered_image_path)[0]

    tile_size = registered_image.chunks[:2]

    roi_h, roi_w = registered_image.shape[:2]

    # initialize zarr stores to store separated images
    image_pyramids = {stain: create_pyramid_group(zarr_store, (roi_h, roi_w), tile_size, np.uint8) for stain, zarr_store in zarr_stores.items()}

    # initialize slices
    slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=tile_size, col_first=False)

    # create uncompressed memory mapped array
    array_synchronizer = zarr.ThreadSynchronizer()
    mem_mapped_array = zarr.open(store=temp_store,
                                 mode="w",
                                 path="uncrompressed",
                                 shape=registered_image.shape,
                                 dtype=registered_image.dtype,
                                 compressor=None,
                                 chunks=registered_image.chunks,
                                 synchronizer=array_synchronizer
                                 )

    mask_synchronizer = zarr.ThreadSynchronizer()
    mem_mapped_mask = zarr.open(store=temp_store,
                                mode="w",
                                path="mask",
                                compressor=None,
                                shape=registered_image.shape[:2],
                                dtype=bool,
                                chunks=registered_image.chunks[:2],
                                synchronizer=mask_synchronizer)

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

    logger.info(f"Decompressed image and calculated threshold {th} in {t_final:.2f} min.")

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

    # create a generator of the deconvolved tiles
    deconvolved_tiles = (deconvolve(slice_tuple,
                                    stain_matrix=stain_matrix,
                                    max_c=max_c,
                                    mmep_array=mem_mapped_array,
                                    mask_array=mem_mapped_mask,
                                    stains=set(zarr_stores.keys()))
                         for slice_tuple in slices)

    zarr_store_paths = {stain: zarr_store.path for stain, zarr_store in zarr_stores.items()}
    synchronizer = zarr.ThreadSynchronizer()
    futures = [executor.submit(partial(save,
                                       tup,
                                       image_pyramids=image_pyramids,
                                       zarr_store_paths=zarr_store_paths,
                                       synchronizer=synchronizer
                                       ))
               for tup in zip(deconvolved_tiles, slices)]

    for _ in as_completed(futures):
        pass

    t_e = time.time()
    t_final = (t_e - t_s) / 60
    logger.info(f"Deconvoluted and saved image in {t_final:.2f} min.")


def save(deconvolved_slices: Tuple[Dict[Stain, np.ndarray], Tuple[slice, slice]],
         image_pyramids: Dict[Stain, Dict[int, str]],
         zarr_store_paths: Dict[Stain, str],
         synchronizer: zarr.sync.ThreadSynchronizer) -> None:
    deconvolved, tile_slices = deconvolved_slices
    for stain, image in deconvolved.items():
        write_slice_to_zarr_location(slice_image=image,
                                     image_pyramid=image_pyramids[stain],
                                     tile_slices=tile_slices,
                                     zarr_store_path=zarr_store_paths[stain],
                                     synchronizer=synchronizer)


def create_pyramid_group(store: zarr.DirectoryStore,
                         shape: Tuple,
                         chunksize: Tuple[int, int],
                         dtype: Type) -> Dict[int, str]:
    """Creates a zarr group for a roi to save an image pyramid
    @param store: directory store for which arrays need to be created
    @param shape: the shape of the array to create
    @param chunksize: the size of the level zero chunk
    @param dtype: the data type stored in the arrays

    """
    root = zarr.group(store)

    pyramid_dict: Dict[int, str] = {}

    h, w = shape[:2]

    for i in range(8):
        chunk_h, chunk_w = chunksize  # taking the slice size aligns chunks, so that multiprocessing only always acessess one chunk
        factor = 2 ** i

        new_h, new_w = int(np.ceil(h / factor)), int(np.ceil(w / factor))

        # if the height, width of the down sampled images is smaller than the chunk size, the height or width is assigned
        if new_h <= chunk_h:
            new_chunk_h = new_h
        else:
            new_chunk_h = chunk_h

        if new_w <= chunk_w:
            new_chunk_w = new_w
        else:
            new_chunk_w = chunk_w

        arr: zarr.Array = root.create(
            str(i),
            shape=(new_h, new_w) + ((shape[2],) if len(shape) == 3 else ()),
            chunks=(new_chunk_h, new_chunk_w) + ((shape[2],) if len(shape) == 3 else ()),
            dtype=dtype,
            overwrite=True,
            compressor=Jpeg(colorspace_data="GRAY", colorspace_jpeg="GRAY", level=75, lossless=False)
        )

        pyramid_dict[i] = arr.path

    return pyramid_dict
