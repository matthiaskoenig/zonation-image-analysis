import matplotlib.cm
import numpy as np
import openslide
import zarr
from matplotlib import pyplot as plt, cm

PATH_TO_PIC = "/home/jkuettner/Pictures/wsi/J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.ndpi"
PATH_TO_ZARR = "/home/jkuettner/Pictures/zarr/img2.zarr"

if __name__ == "__main__":
    image = openslide.OpenSlide(PATH_TO_PIC)
    mask = zarr.open(store=PATH_TO_ZARR, mode="r+")

    x = 54064 - 256
    y = 50547 - 256

    region = image.read_region((x, y), 0, (512, 512))

    selection = mask.get_basic_selection((slice(y, y + 512), slice(x, x + 512)))
    print(np.max(selection))
    selection = np.ma.masked_where(selection == 0, selection)
    fig, (ax, ax1) = plt.subplots(1, 2)
    ax: plt.Axes
    ax.imshow(region)
    ax1.imshow(region)
    ax.imshow(selection, cmap=matplotlib.colormaps.get_cmap("jet"))
    plt.show()
