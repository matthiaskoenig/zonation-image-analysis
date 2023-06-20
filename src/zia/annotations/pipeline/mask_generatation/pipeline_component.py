import cv2

from zia.annotations.annotation.annotations import AnnotationParser
from zia.annotations.annotation.roi import PyramidalLevel
from zia.annotations.path_utils.path_util import FileManager, ResultDir
from zia.annotations.pipeline.mask_generatation.image_analysis import MaskGenerator
from zia.annotations.pipeline.pipeline import IPipelineComponent
from zia.annotations.zarr_image.image_repository import ImageRepository
from zia.annotations.zarr_image.zarr_image import ZarrGroups, ZarrImage


class MaskCreationComponent(IPipelineComponent):
    def __init__(
        self,
        file_manager: FileManager,
        image_repo: ImageRepository,
        draw=True,
        overwrite=False,
    ):
        IPipelineComponent.__init__(self, file_manager, image_repo)
        self._draw = draw
        self._overwrite = overwrite

    def run(self):
        for species, image_name in self._file_manager.get_image_names():
            print(species, image_name)
            zarr_image = self._image_repo.zarr_images.get(image_name)
            if self._check_if_exists(zarr_image) & ~self._overwrite:
                continue

            annotations = AnnotationParser.parse_geojson(
                self._file_manager.get_annotation_path(image_name)
            )
            MaskGenerator.create_mask(zarr_image, annotations)
            self._draw_mask(zarr_image, species)

    def _draw_mask(self, zarr_image: ZarrImage, species: str):
        if not self._draw:
            return

        for i, (leveled_roi, _) in enumerate(zarr_image.iter_rois()):
            image_7, _ = leveled_roi.get_down_sized_level(PyramidalLevel.SEVEN)
            mask = zarr_image.get_liver_mask(i, PyramidalLevel.SEVEN)

            image_7[~mask] = [255, 255, 255]

            cv2.imwrite(
                self._file_manager.get_report_path(
                    ResultDir.LIVER_MASK, species, f"{zarr_image.name}.jpeg"
                ),
                image_7,
            )

    def _check_if_exists(self, zarr_image: ZarrImage) -> bool:
        if ZarrGroups.LIVER_MASK.value in zarr_image.data.keys():
            return True
        else:
            return False


if __name__ == "__main__":
    file_manager = FileManager()
    image_repo = ImageRepository(file_manager)
    mask_generator = MaskCreationComponent(file_manager, image_repo, False)
    mask_generator.run()
