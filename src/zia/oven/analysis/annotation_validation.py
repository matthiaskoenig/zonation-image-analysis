from zia import BASE_PATH
from zia.pipeline.annotation import AnnotationParser, AnnotationType
from zia.oven.data_store import DataStore
from zia.log import get_logger
from zia.oven.path_utils import FileManager

logger = get_logger(__name__)

if __name__ == "__main__":
    from zia.config import read_config

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=None
    )

    annotation_dict = {}

    for image_info in file_manager.get_images():
        data_store = DataStore(image_info=image_info)

        annotations = AnnotationParser.parse_geojson(data_store.image_info.annotations_path)

        artifact_annotations = AnnotationParser.get_annotation_by_types(annotations, AnnotationType.get_artifacts())

        for anno in annotations:
            geom_tyep = str(anno.geometry.geom_type)

            if not geom_tyep in annotation_dict.keys():
                annotation_dict[geom_tyep] = []

            annotation_dict[geom_tyep].append(image_info.metadata.image_id)

    if "LineString" in annotation_dict.keys():
        image_ids = "\n".join(k for k in annotation_dict["LineString"])
        logger.error(f"Line String in artifacts annotations:\n{image_ids}")
    else:
        logger.info("Annotations are fine")
