from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from zia import BASE_PATH
from zia.io.wsi_tifffile import read_ndpi
from zia.pipeline.pipeline_components.algorithm.droplet_detection.droplet_detection import get_foreground_mask

sample_data = [
    ("FLR-168", 0, [(16550, 29629), (12035, 19231)]),
    ("FLR-171", 0, [(12569, 26993), (17931, 37453)]),
    ("FLR-180", 0, [(13492, 24716), (7229, 13357)]),
    ("FLR-200", 0, [(6117, 11060), (15464, 14639)]),
    ("MNT-032", 0, [(10871, 52349), (12939, 50655)]),
    ("MNT-031", 1, [(8936, 10535), (8577, 5367)]),
    ("MNT-043", 1, [(11553, 20721), (14921, 23674)]),
    ("MNT-045", 0, [(5076, 28881), (8511, 33915)]),
    # ["UKJ-19-015_Human", 0, [[37422, 47631], [36319, 28667]]],
    # ["UKJ-19-054_Human", 0, [[24453, 32857], [32875, 28183]]]
]

control_data = [
    ("MNT-023", 0, [(12027, 25623), (8890, 47559)]),
    ("MNT-026", 0, [(7434, 30186), (13453, 12340)]),
    ("NOR-022", 0, [(14350, 33054), (19157, 22806)]),
    ("NOR-025", 0, [(25562, 15182), (22869, 33751)]),
]


def plot(images: list[np.ndarray], path: Path):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4 * len(images), 4), dpi=300)

    split = path.stem.split("_")

    subject, roi = split[:2]

    cmap = None
    for ax, img in zip(axes, images):
        if len(img.shape) == 2:
            cmap = "binary"
        ax.imshow(img, aspect="equal", cmap=cmap)

    fig.suptitle(f"subject: {subject}, roi: {roi}")

    plt.savefig(path)
    plt.close(fig)
    # plt.show()


def create_data(sample_data: List[Tuple[str, int, List[Tuple[int, int]]]], paths: Dict[str, Path], control=False) -> None:
    project = "steatosis" if not control else "control"
    for subject, roi, positions in sample_data:
        path = Path(f"D:/image_data/{project}/RoiExtraction/{subject}/{roi}")

        image_file = None
        for p in path.iterdir():
            if p.is_file() and "HE" in p.stem:
                image_file = p
                break

        array = read_ndpi(image_file)[0]

        suffix = "sample"
        if control:
            suffix = "control"

        for i, position in enumerate(positions):
            h, w = position
            sub_array = array[h: h + 2000, w: w + 2000]

            sub_array = cv2.cvtColor(sub_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(paths["image"] / f"{subject}_{roi}_{suffix}_{i}.png"), sub_array)

            th = get_foreground_mask(sub_array)

            cv2.imwrite(str(paths["mask"] / f"{subject}_{roi}_{suffix}_{i}.png"), th)

            overlay = sub_array.copy()
            contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(overlay, contours, -1, color=(232, 235, 52), thickness=2)
            cv2.imwrite(str(paths["overlay"] / f"{subject}_{roi}_{suffix}_{i}.png"), overlay)

            plot([sub_array, th, overlay], paths["plot"] / f"{subject}_{roi}_{suffix}_{i}.png")


if __name__ == "__main__":
    result_dir = BASE_PATH / "sample_data"
    result_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "mask": result_dir / "background_mask",
        "image": result_dir / "he_image",
        "overlay": result_dir / "overlay_image",
        "plot": result_dir / "combined",
    }

    print(result_dir)

    for p in paths.values():
        p.mkdir(exist_ok=True, parents=True)

    create_data(sample_data=sample_data, paths=paths)
    create_data(sample_data=control_data, paths=paths, control=True)
