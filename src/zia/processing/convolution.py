import matplotlib.pyplot as plt
import numpy as np
import zarr

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.config import read_config
from zia.data_store import ZarrGroups

subject = "NOR-021"
roi = "0"
level = PyramidalLevel.SIX
if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")

    zarr_store = zarr.open(store=config.image_data_path / "stain_separated" / f"{subject}.zarr")
    group = zarr_store.get(f"{ZarrGroups.STAIN_1.value}/{roi}")

    arrays = {}
    for i, a in group.items():
        arrays[i] = np.array(a.get(f"{level}"))

    fig, axes = plt.subplots(2, 3, figsize=(16, 6), dpi=600)

    for key, ax in zip(arrays, axes.flatten()):
        print(arrays[key].shape)
        ax.imshow(arrays[key], cmap="binary_r")
        ax: plt.Axes
        ax.set_title(key)

    plt.show()


