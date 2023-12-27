from pathlib import Path
from typing import Dict, List

import numpy as np
import zarr

from zia.pipeline.common.roi import Roi
from zia.pipeline.common.slicing import get_final_slices, get_tile_slices
from zia.pipeline.common.resolution_levels import PyramidalLevel
from zia.pipeline.file_management.file_management import Slide, SlideFileManager
from zia.pipeline.pipeline_components.pipeline import IPipelineComponent
from zia.pipeline.common.project_config import Configuration
from zia.pipeline.pipeline_components.roi_detection_component import RoiDetectionComponent
from zia.pipeline.common.tile_generator import TileGenerator
from zia.io.wsi_tifffile import read_ndpi, write_rois_to_ome_tiff
from zia.log import get_logger

log = get_logger(__file__)


def extract_roi(arrays: List[zarr.Array], roi: Roi, roi_wsi_path: Path):
    zero_array = arrays[0]
    tile_size_log = 12

    # create generator for each roi and each resolution levels

    level_generator_dict: Dict[PyramidalLevel, TileGenerator] = {}
    for array in arrays:
        level = np.round(np.log2(zero_array.shape[0] / array.shape[0]))
        pyramidal_level = PyramidalLevel.get_by_numeric_level(int(level))
        tile_size = 2 ** (tile_size_log - pyramidal_level)

        roi_cs, roi_rs = roi.get_bound(pyramidal_level)
        roi_polygon = roi.get_polygon_for_level(pyramidal_level)
        roi_h, roi_w = roi_rs.stop - roi_rs.start, roi_cs.stop - roi_cs.start

        # initialize slices
        slices = get_tile_slices(shape=(roi_h, roi_w), tile_size=(tile_size, tile_size),
                                 col_first=False)
        final_slices = get_final_slices((roi_rs, roi_cs), slices)

        level_generator_dict[pyramidal_level] = TileGenerator(
            slices=final_slices,
            array=array,
            roi_shape=(roi_h, roi_w, 3),
            tile_size=tile_size,
            roi_polygon=roi_polygon)

    write_rois_to_ome_tiff(roi_wsi_path, level_generator_dict)


class RoiExtractionComponent(IPipelineComponent):
    """Pipeline step for ROI processing."""
    dir_name = "RoiExtraction"

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, overwrite: bool = False):
        super().__init__(project_config, file_manager, RoiExtractionComponent.dir_name, overwrite)

    def run(self) -> None:
        for slide in self.file_manager.slides:
            self.extract_rois(slide)

    def extract_rois(self, slide) -> None:
        rois = Roi.load_from_file(self.get_roi_path(slide))

        arrays = read_ndpi(self.project_config.data_path / slide.species / slide.protein.upper() / f"{slide.name}.{self.project_config.extension}")

        for roi in rois:
            roi_wsi_path = self.get_roi_wsi(slide, roi)

            if self.check_overwrite(roi_wsi_path, slide, roi):
                continue

            log.info(f"[{slide.subject}\t{slide.protein}]\tExtracting and writing ROI {roi.lobe}")
            extract_roi(arrays, roi, roi_wsi_path)

    def get_roi_path(self, slide: Slide) -> Path:
        p = self.project_config.image_data_path / RoiDetectionComponent.dir_name / f"{slide.name}.geojson"
        if not p.exists():
            raise FileNotFoundError(f"No Roi geojson file found for {slide.name}")
        return p

    def get_roi_wsi(self, slide: Slide, roi: Roi) -> Path:
        file_dir = self.image_data_path / slide.subject / roi.lobe
        file_dir.mkdir(parents=True, exist_ok=True)
        return file_dir / f"{slide.name}.ome.tiff"

    def check_overwrite(self, roi_wsi_path, slide: Slide, roi: Roi):

        if not self.overwrite and roi_wsi_path.exists():
            log.info(f"Tiff file for subject: {slide.subject}, protein: {slide.protein}, roi: {roi.lobe} already exists."
                     f" Set overwrite True to overwrite.")
            return True
        return False
