"""Plotting helpers for images."""
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 25


def plot_image_with_hist(image: np.ndarray, cmap: str = "gray", title: str = None, path: Path=None):
    """Plot the relevant channels."""
    plt.figure(figsize=(20, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    print("CYP2E1")
    plt.subplot(221)
    plt.imshow(image[:, :, 0], cmap=cmap, aspect="equal")
    # plt.imshow(data[:, :, 2], cmap=cmap, aspect="equal", alpha=0.5)
    plt.axis('off')
    plt.title("CYP2E1")

    plt.subplot(223)
    hist, bins = np.histogram(image[:, :, 0], bins=256)  # , 256, [0,256])
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
    plt.title("CYP2E1 Histogram")
    print(hist)

    # ECAD
    print("ECAD")
    plt.subplot(222)
    plt.imshow(image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis('off')
    plt.title("E-Cadherin")

    plt.subplot(224)
    hist, bins = np.histogram(image[:, :, 1], bins=256)  # , 256, [0,256])
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
    plt.title("E-Cadherin Histogram")
    print(hist)

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()


def plot_zonation(image: np.ndarray, cmap="gray", title: str = None, path: Path=None):
    """Plot the relevant channels."""
    plt.figure(figsize=(20, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    print("CYP2E1")
    plt.subplot(221)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.axis('off')
    plt.title("CYP2E1")

    plt.subplot(222)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.axis('off')
    plt.title("E-Cadherin")

    # ECAD
    print("CYP2E1-ECAD")
    plt.subplot(223)
    plt.imshow(image[:, :, 0] - image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis('off')
    plt.title("CYP2E1 - E-Cadherin")

    print("CYP2E1/ECAD")
    plt.subplot(224)
    plt.imshow(image[:, :, 0]/image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis('off')
    plt.title("CYP2E1/E-Cadherin")

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()


def plot_overlay(image: np.ndarray, image_gauss, cmap="gray", alpha=0.5, title: str = None, path: Path=None):
    """Plot the relevant channels."""
    X, Y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
    image_difference = image_gauss[:, :, 0] - image_gauss[:, :, 1]
    image_ratio = image_gauss[:, :, 0]/image_gauss[:, :, 1]

    plt.figure(figsize=(20, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    plt.subplot(221)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.imshow(image_difference, cmap=cmap, aspect="equal", alpha=alpha)
    plt.contour(X, Y, image_difference, colors="black", linestyles="solid")
    plt.axis('off')
    plt.title("CYP2E1 + Difference Overlay")

    plt.subplot(222)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.imshow(image_difference, cmap=cmap, aspect="equal", alpha=alpha)
    plt.contour(X, Y, image_difference, colors="black", linestyles="solid")
    plt.axis('off')
    plt.title("E-Cadherin + Difference Overlay")

    plt.subplot(223)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.imshow(image_ratio, cmap=cmap, aspect="equal", alpha=alpha)
    plt.contour(X, Y, image_ratio, colors="black", linestyles="solid")
    plt.axis('off')
    plt.title("CYP2E1 + Ratio Overlay")

    plt.subplot(224)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.imshow(image_ratio, cmap=cmap, aspect="equal",
               alpha=alpha)
    plt.contour(X, Y, image_ratio, colors="black", linestyles="solid")
    plt.axis('off')
    plt.title("E-Cadherin + Ratio Overlay")

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()

