import os
import logging
from typing import List

from PIL.ImageDraw import ImageDraw
from PIL.Image import Image
from openslide import OpenSlide
from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.annotation.geometry_utils import read_full_image_from_slide
from zia.annotations.annotation.roi import Roi, PyramidalLevel
from zia.annotations.pipeline.roi_segmentation import RoiSegmentation

IMAGE_PATH = "/home/jkuettner/Pictures/wsi_data/control/rat"
ANNO_PATH = "/home/jkuettner/Pictures/wsi_annotations/annotations_species_comparison/rat_project/objectsjson"
RESULT_PATH = "/home/jkuettner/Pictures/wsi_annotations/annotations_liver_roi"
REPORT_PATH = "/home/jkuettner/Development/git/zonation-image-analysis/results/liver_rois"

logger = logging.getLogger(__name__)


class RoiSegmentationReport:
    total = 0
    annotation_geojson_missing = []
    liver_annotation_missing = []
    segmentation_success = []
    segmentation_fail = []

    def register_geojson_missing(self, file):
        self.annotation_geojson_missing.append(file)
        self.total += 1

    def register_liver_annotation_missing(self, file):
        self.liver_annotation_missing.append(file)
        self.total += 1

    def register_segmentation_fail(self, file):
        self.segmentation_fail.append(file)
        self.total += 1

    def register_segmentation_success(self, file):
        self.segmentation_success.append(file)
        self.total += 1

    def _get_list_as_string(self, l: List[str]) -> str:
        return "\n".join(l)

    def report(self):
        assert len(self.segmentation_fail) + len(self.segmentation_success) + len(
            self.annotation_geojson_missing) + len(
            self.liver_annotation_missing) == self.total

        result = f"Report:\n"
        result += 80 * "-" + "\n"
        result += f"Total files processed: {self.total}\n"
        result += f"Annotation geojson missing: {len(self.annotation_geojson_missing)}\n"
        result += f"Liver annotation missing: {len(self.liver_annotation_missing)}\n"
        result += f"Segmentation Success: {len(self.segmentation_success)}\n"
        result += f"Segmentation Fail: {len(self.segmentation_fail)}\n"
        result += 80 * "-" + "\n"
        result += f"Annotation geojson files missing for: \n{self._get_list_as_string(self.annotation_geojson_missing)}\n"
        result += 50 * "-" + "\n"
        result += f"Liver annotation missing for: \n{self._get_list_as_string(self.liver_annotation_missing)}"

        return result

    def __str__(self):
        return self.report()

    def save(self, file):
        with open(file, "w") as f:
            f.write(self.__str__())


def draw_result(open_slide: OpenSlide, liver_roi: Roi) -> Image:
    region = read_full_image_from_slide(open_slide, 7)
    draw = ImageDraw(region)
    poly_points = liver_roi.get_polygon_for_level(PyramidalLevel.SEVEN).exterior.coords
    draw.polygon(list(poly_points), outline="red", width=3)
    return region


if __name__ == "__main__":
    report = RoiSegmentationReport()
    for folder_name in os.listdir(IMAGE_PATH):
        folder_path = os.path.join(IMAGE_PATH, folder_name)

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            geojson_file = os.path.splitext(file_name)[0] + ".geojson"

            if geojson_file not in os.listdir(ANNO_PATH):
                logger.warning(
                    "No annotation geojson file found for '" + file_name + "'.")
                report.register_geojson_missing(file_name)
                continue

            annotations = AnnotationParser.parse_geojson(
                os.path.join(ANNO_PATH, geojson_file))

            liver_annotations = AnnotationParser.get_annotation_by_type(annotations,
                                                                        AnnotationType.LIVER)

            if len(liver_annotations) == 0:
                report.register_liver_annotation_missing(file_name)
                continue

            if len(liver_annotations) > 1:
                logger.warning(
                    "Multiple annotations found for liver for file '" + file_name + "'.")

            open_slide = OpenSlide(os.path.join(folder_path, file_name))
            liver_roi = RoiSegmentation.find_rois(open_slide, annotations,
                                                  AnnotationType.LIVER)

            if liver_roi is None:
                logger.warning("No ROI found for '" + file_name + "'.")
                report.register_segmentation_fail(file_name)

            else:
                liver_roi.write_to_geojson(os.path.join(RESULT_PATH, geojson_file))
                report.register_segmentation_success(file_name)
                image = draw_result(open_slide, liver_roi)
                image.save(
                    os.path.join(REPORT_PATH, os.path.splitext(file_name)[0] + ".png"),
                    "PNG")

    print(report)
    report.save(os.path.join(REPORT_PATH, "report.txt"))
