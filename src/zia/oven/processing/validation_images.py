from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from zia import BASE_PATH
from zia.config import read_config
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats
from imagecodecs.numcodecs import Jpeg2k, Jpeg
import numcodecs

numcodecs.register_codec(Jpeg2k)
numcodecs.register_codec(Jpeg)


def find_cyp2e1_image(roi_image_path: Path):
    for p in roi_image_path.iterdir():
        if "CYP2E1" in p.stem:
            return p
    raise FileNotFoundError("No image for CYP2E1 found.")


if __name__ == "__main__":

    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "lobule_segmentation"
    report_path.mkdir(parents=True, exist_ok=True)
    data_dir_stain_separated = config.image_data_path / "slide_statistics"

    subject_dirs = sorted([f for f in data_dir_stain_separated.iterdir() if f.is_dir() and not f.name.startswith(".")])

    for subject_dir in subject_dirs:
        subject = subject_dir.stem
        roi_dict = {}

        # print(subject)

        roi_dirs = sorted([f for f in subject_dir.iterdir() if f.is_dir()])
        for roi_dir in roi_dirs:
            roi = roi_dir.stem
            print(subject, roi)
            # print(roi)
            slide_stats = SlideStats.load_from_file_system(roi_dir)
            zarr_store = config.image_data_path / "stain_separated" / f"{subject}.zarr"
            roi_protein = np.array(zarr.open(store=zarr_store, path=f"stain_1/{roi}/CYP2E1/7"))

            fig, ax = plt.subplots(1, 1, dpi=300)
            ax: plt.Axes
            ax.imshow(roi_protein, cmap="binary_r")

            slide_stats.plot_on_axis(ax,
                                     lobulus_fc="lime",
                                     lobulus_alpha=0.2)
            plt.savefig(report_path / f"{subject}_{roi}.jpeg")

            plt.show()
