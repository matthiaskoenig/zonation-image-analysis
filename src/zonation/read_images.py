"""Reading images.

Using bioformats with python-javabridge

https://pypi.org/project/python-javabridge/
https://pythonhosted.org/javabridge/installation.html

git clone https://github.com/CellProfiler/python-javabridge.git

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

"""
from pathlib import Path

import numpy as np
from rich.console import Console
console = Console()

import javabridge
import bioformats
from matplotlib import pyplot as plt


def read_czi(path: Path) -> np.ndarray:
    """Read czi_data."""

    javabridge.start_vm(class_path=bioformats.JARS)

    with bioformats.ImageReader(str(path)) as reader:
        data = reader.read()
        print(type(data))
        print(data.shape)

    # metadata = bioformats.get_omexml_metadata(czi_path)
    # console.print(metadata)

    javabridge.kill_vm()
    return data


def min_max_normalization(image: np.ndarray):
    return (image-np.min(image))/(np.max(image)-np.min(image))




class Image:
    """Create image and serialize."""


    def __init__(self, channel_names, data):
        pass


def histogram_normalization():
    """Remove outliers at high values & rescale histogram to 256 range."""
    pass

def plot_histogram(data: np.ndarray):
    i = 0
    data_flat = data[..., i].flatten()
    print(data_flat.max())
    hist, bins = np.histogram(data_flat, bins=256) # , 256, [0,256])

    print(hist)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    bin_width = bins[1] - bins[0]
    # plt.plot(cdf_normalized, label="cdf")
    plt.bar(bins[:-1], hist*bin_width, bin_width, color="black")

    plt.xscale = "log"
    # plt.xlim([0, 256])
    plt.show()


def gauss_filter():
    """Smoothening for processing."""
    pass

def watershed():
    pass

def zonation_position():
    pass


def plot_image(data: np.ndarray, cmap: str = "gray"):
    """Plot the relevant channels."""
    plt.figure(figsize=(24, 8))

    # E-catharin
    plt.subplot(131)
    plt.imshow(data[:, :, 0], cmap=cmap, aspect="equal", alpha=1.0)
    # plt.imshow(data[:, :, 2], cmap=cmap, aspect="equal", alpha=0.5)
    plt.axis('off')
    plt.title("ecad")

    # CYP2E1
    plt.subplot(132)
    plt.imshow(data[:, :, 2], cmap=cmap)
    plt.axis('off')
    plt.title("cyp2e1")

    # Zonation
    plt.subplot(133)
    plt.imshow(data[:, :, 2]/data[:, :, 0], cmap=cmap)
    plt.axis('off')
    plt.title("cyp2e1/ecad")
    plt.show()


if __name__ == "__main__":
    from zonation import CZI_PATH
    # data = read_czi(CZI_PATH / "M1039R_4x5_1.czi")
    data = read_czi(CZI_PATH / "RH0422_1_4x5.czi")
    plot_image(data, cmap="tab20c")
    plot_histogram(data)