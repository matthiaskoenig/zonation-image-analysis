import numpy as np
from PIL import Image

from zia import BASE_PATH
from zia.data_store import DataStore, ZarrGroups
from zia.path_utils import FileManager, ResultsDirectories, filter_factory

from zia.pipeline.annotation import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_rgb

if __name__ == "__main__":
    from zia.config import read_config

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=filter_factory(species="rat", subject="NOR-025", protein="cyp1a2")
    )

    # set the level for which the image should be created. Everything smaller than 4
    # gets memory intense
    level = PyramidalLevel.FIVE

    results_path = file_manager.results_path / ResultsDirectories.MASKED_PNG_IMAGES.value
    results_path.mkdir(parents=True, exist_ok=True)

    for image_info in file_manager.get_images():
        data_store = DataStore(image_info)
        image_id = image_info.metadata.image_id
        for i, roi in enumerate(data_store.rois):
            mask_array = data_store.get_array(ZarrGroups.LIVER_MASK, i, level)

            roi_image = data_store.read_full_roi(i, level)
            image_array = np.array(roi_image)
            image_array = image_array[:, :, :-1]

            image_array[~mask_array[:]] = (255, 255, 255)

            masked_image = Image.fromarray(image_array)

            plot_rgb(masked_image, transform_to_bgr=False)

            masked_image.save(
                file_manager.results_path / ResultsDirectories.MASKED_PNG_IMAGES.value / f"{image_id}.png")
