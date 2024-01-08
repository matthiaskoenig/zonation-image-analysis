import re
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tifffile import imwrite

from zia.oven.annotations.pipelines.stain_separation.stain_separation_whole_image import separate_raw_image
from zia.io.wsi_tifffile import read_ndpi

if __name__ == "__main__":

    files = ["MNT-025_CYP1A2.ome.tiff",
             "MNT-025_CYP2D6.ome.tiff",
             "MNT-025_CYP2E1.ome.tiff",
             "MNT-025_CYP3A4.ome.tiff",
             "MNT-025_GS.ome.tiff",
             "MNT-025_HE.ome.tiff"]

    base_path = Path("/home/jkuettner/Downloads/mouse_registered_example/mouse")
    results_path_stain_0 = Path("/home/jkuettner/Downloads/result/stain_0")
    results_path_stain_1 = Path("/home/jkuettner/Downloads/result/stain_1")

    results_path_stain_0.mkdir(exist_ok=True)
    results_path_stain_1.mkdir(exist_ok=True)

    base_size = 3
    fig, axes = plt.subplots(2, len(files), figsize=(3 * len(files), 3 * 0.90 * 2), dpi=1000)
    axes[0, 0].set_ylabel("Hematoxylin")
    axes[0, 1].set_ylabel("DAB/Eosin")
    fig.suptitle(re.split("_", files[0])[0])
    for i, file in enumerate(files):
        protein = re.split("_|\.", file)[1]
        print(protein)
        image_array = np.array(read_ndpi(base_path / file)[0])
        stain_0, stain_1 = separate_raw_image(image_array)
        axes[0, i].imshow(stain_0, vmin=0, vmax=255, cmap="binary_r")
        axes[1, i].imshow(stain_1, vmin=0, vmax=255, cmap="binary_r")

        axes[0, i].set_title(protein)
        imwrite(results_path_stain_0 / file, stain_0, photometric="MINISBLACK")
        imwrite(results_path_stain_1 / file, stain_1, photometric="MINISBLACK")


    for ax in axes.flatten():
        ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.savefig(f"/home/jkuettner/Downloads/aligned_and_separated.png")

    plt.show()
