from pathlib import Path

import cv2

from zia.io.wsi_tifffile import read_ndpi
from zia.pipeline.pipeline_components.algorithm.droplet_detection.droplet_detection import get_foreground_mask

sample_data = [
    ["FLR-168", 0, [[16550, 29629], [12035, 19231]]],
    ["FLR-171", 0, [[12569, 26993], [17931, 37453]]],
    ["FLR-180", 0, [[13492, 24716], [7229, 13357]]],
    ["FLR-200", 0, [[6117, 11060], [15464, 14639]]],
    ["MNT-032", 0, [[10871, 52349], [12939, 50655]]],
    ["MNT-031", 1, [[8936, 10535], [8577, 5367]]],
    ["MNT-043", 1, [[11553, 20721], [14921, 23674]]],
    ["MNT-045", 0, [[5076, 28881], [8511, 33915]]],
    #["UKJ-19-015_Human", 0, [[37422, 47631], [36319, 28667]]],
    #["UKJ-19-054_Human", 0, [[24453, 32857], [32875, 28183]]]
]


if __name__ == "__main__":
    result_dir = Path(r"C:\Users\jonas\Development\git\zonation-image-analysis\data_set")
    mask_dir = result_dir / "background_mask"
    image_dir = result_dir / "he_image"

    for p in [result_dir, mask_dir, image_dir]:
        p.mkdir(exist_ok=True, parents=True)

    for subject, roi, positions in sample_data:
        path = Path(f"D:/image_data/steatosis/RoiExtraction/{subject}/{roi}")

        image_file = None
        for p in path.iterdir():
            if p.is_file() and "HE" in p.stem:
                image_file = p
                break

        array = read_ndpi(image_file)[0]

        for i, position in enumerate(positions):
            h, w = position
            sub_array = array[h: h + 2000, w: w + 2000]

            #cv2.cvtColor(sub_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_dir / f"{subject}_{roi}_sample_{i}.png"), sub_array)


            th = get_foreground_mask(sub_array)

            cv2.imwrite(str(mask_dir / f"{subject}_{roi}_sample_{i}.png"), th)

