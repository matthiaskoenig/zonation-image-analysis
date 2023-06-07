import os
import logging
import time
from typing import List

from PIL.ImageDraw import ImageDraw
from PIL.Image import Image

from zia import DATA_PATH, RESULTS_PATH, REPORT_PATH
from zia.annotations import OPENSLIDE_PATH
from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.annotation.geometry_utils import read_full_image_from_slide
from zia.annotations.annotation.roi import Roi, PyramidalLevel
from zia.annotations.pipeline.roi_segmentation import RoiSegmentation

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

IMAGE_PATH = DATA_PATH / "cyp_species_comparison" / "control"
ANNO_PATH = DATA_PATH / "annotations_species_comparison"
ROI_RESULT_PATH = RESULTS_PATH / "annotations_liver_roi"
ROI_REPORT_PATH = REPORT_PATH / "liver_rois"

for p in [ROI_RESULT_PATH, ROI_REPORT_PATH]:
    p.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

species_list = ["rat", "mouse", "human", "pig"]


def get_annotation_path(species_name: str):
    return os.path.join(ANNO_PATH, f"{species_name}_project", "objectsjson")


def create_paths(base_path):
    for species_name in species_list:
        if not os.path.exists(os.path.join(base_path, species_name)):
            os.makedirs(os.path.join(base_path, species_name))


class RoiSegmentationReport:
    def __init__(self):
        self._total = 0
        self._annotation_geojson_missing = []
        self._liver_annotation_missing = []
        self._segmentation_success = []
        self._segmentation_fail = []
        self._segmentation_partial = []
        self._time = None
        pass

    def register_geojson_missing(self, file):
        self._annotation_geojson_missing.append(file)
        self._total += 1

    def register_liver_annotation_missing(self, file):
        self._liver_annotation_missing.append(file)
        self._total += 1

    def register_segmentation_fail(self, file):
        self._segmentation_fail.append(file)
        self._total += 1

    def register_segmentation_success(self, file):
        self._segmentation_success.append(file)
        self._total += 1

    def register_segmentation_partial(self, file):
        self._segmentation_partial.append(file)
        self._total += 1

    def _get_list_as_string(self, l: List[str]) -> str:
        return "\n".join(l)

    def set_time(self, time: float):
        self._time = time

    def report(self):
        # assert len(self.segmentation_fail) + len(self.segmentation_success) + len(
        #    self.annotation_geojson_missing) + len(
        #    self.liver_annotation_missing) == self.total

        result = f"Report:\n"
        result += 80 * "-" + "\n"
        result += f"Total of {self._total}  files processed in {self._time:.2f} seconds:\n"
        result += f"Annotation geojson missing: {len(self._annotation_geojson_missing)}\n"
        result += f"Liver annotation missing: {len(self._liver_annotation_missing)}\n"
        result += f"Segmentation success: {len(self._segmentation_success)}\n"
        result += f"Segmentation partial success: {len(self._segmentation_partial)}\n"
        result += f"Segmentation Fail: {len(self._segmentation_fail)}\n"
        result += 80 * "-" + "\n"
        result += f"Annotation geojson files missing for: \n{self._get_list_as_string(self._annotation_geojson_missing)}\n"
        result += 50 * "-" + "\n"
        result += f"Liver annotation missing for: \n{self._get_list_as_string(self._liver_annotation_missing)}\n"
        result += 50 * "-" + "\n"
        result += f"Partial Success for: \n{self._get_list_as_string(self._segmentation_partial)}"

        return result

    def __str__(self):
        return self.report()

    def save(self, file):
        with open(file, "w") as f:
            f.write(self.__str__())


def draw_result(open_slide: openslide.OpenSlide, liver_rois: List[Roi]) -> Image:
    region = read_full_image_from_slide(open_slide, 7)
    draw = ImageDraw(region)
    for liver_roi in liver_rois:
        poly_points = liver_roi.get_polygon_for_level(
            PyramidalLevel.SEVEN).exterior.coords
        draw.polygon(list(poly_points), outline="red", width=3)
    return region


if __name__ == "__main__":
    create_paths(ROI_RESULT_PATH)
    create_paths(ROI_REPORT_PATH)
    for species in species_list:
        logger.info(f"Start finding liver ROIs for {species}")
        start_time = time.time()
        report = RoiSegmentationReport()
        image_path = os.path.join(IMAGE_PATH, species)
        for folder_name in os.listdir(image_path):
            folder_path = os.path.join(image_path, folder_name)

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                geojson_file = os.path.splitext(file_name)[0] + ".geojson"

                if geojson_file not in os.listdir(get_annotation_path(species)):
                    logger.warning(
                        "No annotation geojson file found for '" + file_name + "'.")
                    report.register_geojson_missing(file_name)
                    continue

                annotations = AnnotationParser.parse_geojson(
                    os.path.join(get_annotation_path(species), geojson_file))

                liver_annotations = AnnotationParser.get_annotation_by_type(annotations,
                                                                            AnnotationType.LIVER)

                if len(liver_annotations) == 0:
                    report.register_liver_annotation_missing(file_name)
                    continue

                open_slide = openslide.OpenSlide(os.path.join(folder_path, file_name))
                liver_rois = RoiSegmentation.find_rois(open_slide, annotations,
                                                       AnnotationType.LIVER)

                if len(liver_rois) == 0:
                    logger.warning("No ROI found for '" + file_name + "'.")
                    report.register_segmentation_fail(file_name)

                else:
                    Roi.write_to_geojson(liver_rois,
                                         os.path.join(ROI_RESULT_PATH, species,
                                                      geojson_file))
                    image = draw_result(open_slide, liver_rois)
                    image.save(os.path.join(ROI_REPORT_PATH, species,
                                            os.path.splitext(file_name)[0] + ".png"),
                               "PNG")

                    if len(liver_rois) == len(liver_annotations):
                        report.register_segmentation_success(file_name)
                    else:
                        report.register_segmentation_partial(file_name)
        end_time = time.time()
        report.set_time(end_time - start_time)
        print(report)
        report.save(os.path.join(ROI_REPORT_PATH, species, "report.txt"))
        logger.info(f"Finished finding liver ROIs for {species}")
