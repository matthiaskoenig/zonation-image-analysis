import matplotlib.pyplot as plt
import numpy as np

from zia import BASE_PATH
from zia.pipeline.annotation import PyramidalLevel
from zia.data_store import DataStore, ZarrGroups
from zia.path_utils import FileManager, ResultsDirectories, filter_factory

if __name__ == "__main__":
    from zia.config import read_config

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=filter_factory(species="rat", subject="NOR-025", protein="cyp1a2")
    )
    stain = ZarrGroups.STAIN_1
    results_path = file_manager.results_path / ResultsDirectories.STAIN_SEPERATED_IMAGES.value / stain
    results_path.mkdir(parents=True, exist_ok=True)

    level = PyramidalLevel.FIVE

    for image_info in file_manager.get_images():
        data_store = DataStore(image_info=image_info)
        for i, roi in enumerate(data_store.rois):
            try:
                if image_info.metadata.protein == "he":
                    continue

                if image_info.metadata.negative:
                    continue

                dab = np.array(data_store.get_array(stain, roi_no=i, level=level))

                fig, ax = plt.subplots(1, 1, dpi=600)
                fig: plt.Figure
                ax: plt.Axes


                ax.imshow(dab, cmap="binary_r")
                fig.suptitle(f"{image_info.metadata.image_id} ROI: {i}")

                plt.savefig(
                    results_path / f"{image_info.metadata.image_id}_{i}.png")

                plt.show()
                plt.close(fig)

            except Exception as e:
                print(data_store.image_info.metadata.image_id, i)
