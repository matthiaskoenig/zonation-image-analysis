import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from zia import BASE_PATH
from zia.data_store import DataStore, ZarrGroups
from zia.path_utils import FileManager, ResultsDirectories, filter_factory

from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_rgb

if __name__ == "__main__":
    from zia.config import read_config

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=None #filter_factory(species="rat", subject="NOR-025", protein="cyp1a2")
    )
    stain = ZarrGroups.DAB_STAIN
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
                dab = np.array(data_store.get_array(stain, roi_no=i,
                                                    level=level))

                #print(data_store.image_info.metadata.image_id)

                # print(dab)
                fig, ax = plt.subplots(1, 1, dpi=300)
                fig: plt.Figure
                ax: plt.Axes

                ax.imshow(dab, cmap="binary_r")
                fig.suptitle(f"{image_info.metadata.image_id} ROI: {i}")

                plt.savefig(
                    results_path / f"{image_info.metadata.image_id}_{i}.png")

                plt.close(fig)
            except Exception as e:

                print(data_store.image_info.metadata.image_id, i)
