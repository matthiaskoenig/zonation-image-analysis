from pathlib import Path

from zia.annotations.annotation.annotations import AnnotationParser, AnnotationType
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.data_store import DataStore, ZarrGroups

from zia.annotations.pipelines.pipeline import IPipelineComponent
from zia.annotations.pipelines.mask_generatation.image_analysis import MaskGenerator
from zia.log import get_logger

logger = get_logger(__name__)


class MaskCreationComponent(IPipelineComponent):
    def __init__(self, overwrite=False, draw=True):
        super().__init__(overwrite)
        self._draw = draw

    def run(self, data_store: DataStore, results_path: Path) -> None:
        image_id = data_store.image_info.metadata.image_id

        # prevent from overwriting data from previous runs during development
        if self._check_if_exists(data_store) & ~self.overwrite:
            logger.info(f"[{image_id}]\tMask already exists. To overwrite, set overwrite to True for {self.__class__.__name__}.")
            return

        logger.info(f"[{image_id}]\tStarted Mask generation.")
        geojson_path = data_store.image_info.annotations_path

        # parse annotations for class
        annotations = AnnotationParser.parse_geojson(
            path=geojson_path
        )

        # filter artifact annotations
        artifact_annotations = AnnotationParser.get_annotation_by_types(
            annotations, AnnotationType.get_artifacts()
        )


        if len(artifact_annotations) == 0:
            logger.info(f"[{image_id}]\tNo artifact annotations found.")

        MaskGenerator.create_mask(data_store, artifact_annotations)

        if self._draw:
            for i in range(len(data_store.rois)):
                mask = data_store.data.get(f"{ZarrGroups.LIVER_MASK.value}/{i}/{0}")
                plot_pic(mask[::32, ::32])

    @classmethod
    def _check_if_exists(cls, data_store: DataStore) -> bool:
        if ZarrGroups.LIVER_MASK.value in data_store.data.keys():
            return True
        else:
            return False
