"""Plotting helpers for images."""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


plt.rcParams["font.size"] = 25


def plot_image_with_hist(
    image: np.ndarray, cmap: str = "gray", title: str = None, path: Path = None
):
    """Plot the relevant channels."""
    plt.figure(figsize=(30, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    print("CYP2E1")
    plt.subplot(231)
    plt.imshow(image[:, :, 0], cmap=cmap, aspect="equal")
    # plt.imshow(data[:, :, 2], cmap=cmap, aspect="equal", alpha=0.5)
    plt.axis("off")
    plt.title("CYP2E1")

    plt.subplot(234)
    hist, bins = np.histogram(image[:, :, 0], bins=256)  # , 256, [0,256])
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
    plt.title("CYP2E1 Histogram")
    print(hist)

    # ECAD
    print("ECAD")
    plt.subplot(232)
    plt.imshow(image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis("off")
    plt.title("E-Cadherin")

    plt.subplot(235)
    hist, bins = np.histogram(image[:, :, 1], bins=256)  # , 256, [0,256])
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
    plt.title("E-Cadherin Histogram")
    print(hist)

    if image.shape[2] == 3:
        # DAPI
        print("DAPI")
        plt.subplot(233)
        plt.imshow(image[:, :, 2], cmap=cmap, aspect="equal")
        plt.axis("off")
        plt.title("DAPI")

        plt.subplot(236)
        hist, bins = np.histogram(image[:, :, 2], bins=256)  # , 256, [0,256])
        bin_width = bins[1] - bins[0]
        plt.bar(bins[:-1], hist * bin_width, bin_width, color="black")
        plt.title("DAPI Histogram")
        print(hist)

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()


def plot_zonation(image: np.ndarray, cmap="gray", title: str = None, path: Path = None):
    """Plot the relevant channels."""
    plt.figure(figsize=(20, 15))
    if title:
        plt.suptitle(title)

    # CYP2E1
    print("CYP2E1")
    plt.subplot(221)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.axis("off")
    plt.title("CYP2E1")

    plt.subplot(222)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.axis("off")
    plt.title("E-Cadherin")

    # ECAD
    print("CYP2E1-ECAD")
    plt.subplot(223)
    plt.imshow(image[:, :, 0] - image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis("off")
    plt.title("CYP2E1 - E-Cadherin")

    print("CYP2E1/ECAD")
    plt.subplot(224)
    plt.imshow(image[:, :, 0] / image[:, :, 1], cmap=cmap, aspect="equal")
    plt.axis("off")
    plt.title("CYP2E1/E-Cadherin")

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()


def plot_overlay(
    image: np.ndarray,
    image_gauss: np.ndarray,
    zonation: np.ndarray,
    zonation_gauss: np.ndarray,
    cmap="gray",
    alpha=0.5,
    title: str = None,
    path: Path = None,
):
    """Plot the relevant channels."""
    X, Y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))

    plt.figure(figsize=(20, 30))
    if title:
        plt.suptitle(title)

    plt.subplot(421)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.axis("off")
    plt.title("CYP2E1")

    plt.subplot(422)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.axis("off")
    plt.title("E-Cadherin")

    plt.subplot(423)
    plt.imshow(zonation, cmap=cmap, aspect="equal")
    plt.axis("off")
    plt.title("Zonation")

    plt.subplot(424)
    plt.imshow(zonation_gauss, cmap=cmap, aspect="equal")
    plt.axis("off")
    plt.title("Zonation (Gauss)")

    plt.subplot(425)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.imshow(zonation, cmap=cmap, aspect="equal", alpha=alpha)
    plt.axis("off")
    plt.title("CYP2E1 + Zonation")

    plt.subplot(426)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.imshow(zonation, cmap=cmap, aspect="equal", alpha=alpha)
    plt.axis("off")
    plt.title("E-Cadherin + Zonation")

    plt.subplot(427)
    plt.imshow(image[:, :, 0], cmap="gray", aspect="equal")
    plt.imshow(zonation_gauss, cmap=cmap, aspect="equal", alpha=alpha)
    plt.contour(X, Y, zonation_gauss, colors="black", linestyles="solid")
    plt.axis("off")
    plt.title("CYP2E1 + Zonation (Gauss)")

    plt.subplot(428)
    plt.imshow(image[:, :, 1], cmap="gray", aspect="equal")
    plt.imshow(zonation_gauss, cmap=cmap, aspect="equal", alpha=alpha)
    plt.contour(X, Y, zonation_gauss, colors="black", linestyles="solid")
    plt.axis("off")
    plt.title("E-Cadherin + Zonation (Gauss)")

    if path:
        plt.savefig(path, bbox_inches="tight")

    plt.show()
