from typing import List


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
