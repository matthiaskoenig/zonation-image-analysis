from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt

from zia.oven.annotations.pipelines.stain_separation.stain_separation_whole_image import separate_raw_image
from zia.console import console
from zia.io.wsi_tifffile import read_ndpi

from pathlib import Path

from zia.slide_alignment.image_manipulation import merge_images


def image_prefixes_from_dir(p_dir: Path, stains: List[str]) -> Dict[str, str]:
    paths = [f for f in p_dir.iterdir() if f.is_file()]
    d: Dict[str, str] = {}
    for stain in stains:
        for p in paths:
            if stain in p.name:
                d[stain] = p.name.split(".")[0]
                break
        else:
            raise ValueError(f"Missing stain: '{stain}'")
    return d


def create_registration_image(subject_dir: Path, stains: List[str], results_dir: Path):
    subject_id: str = subject_dir.name

    non_rigid_dir = subject_dir / subject_id / "non_rigid_registration"
    image_prefixes: Dict[str, str] = image_prefixes_from_dir(non_rigid_dir, stains=stains)
    console.log_pair(subject_id)
    console.log_pair(image_prefixes)

    # output_path = subject_dir / f"{subject_id}_registered.png"
    output_path = results_dir / f"{subject_id}_registered.png"
    image_paths = [non_rigid_dir / f"{prefix}.png" for prefix in image_prefixes.values()]
    merge_images(paths=image_paths, output_path=output_path, direction="horizontal")
    console.log_pair(output_path)


def create_stain_separation_image(subject_dir: Path, stains: List[str], results_dir: Path):
    subject_id: str = subject_dir.name

    ome_tiff_dir = subject_dir
    image_prefixes: Dict[str, str] = image_prefixes_from_dir(ome_tiff_dir, stains=stains)
    files = [ome_tiff_dir / f"{prefix}.ome.tiff" for prefix in image_prefixes.values()]
    output_path = results_dir / f"{subject_id}_registered_stain_separated.png"

    console.log_pair(subject_id)
    console.log_pair(image_prefixes)

    console.log_pair(output_path)

    fig, axes = plt.subplots(2, len(files), figsize=(3 * len(files), 3 * 0.90 * 2), dpi=1000)
    axes[0, 0].set_ylabel("Hematoxylin")
    axes[0, 1].set_ylabel("DAB/Eosin")
    fig.suptitle(subject_id)
    for i, file in enumerate(files):
        protein = list(image_prefixes.keys())[i]
        console.print(f"{subject_id}_{protein}")

        image_array = np.array(read_ndpi(file)[0])
        stain_0, stain_1 = separate_raw_image(image_array)
        axes[0, i].imshow(stain_0, vmin=0, vmax=255, cmap="binary_r")
        axes[1, i].imshow(stain_1, vmin=0, vmax=255, cmap="binary_r")

        axes[0, i].set_title(protein)
        # imwrite(results_path_stain_0 / file, stain_0, photometric="MINISBLACK")
        # imwrite(results_path_stain_1 / file, stain_1, photometric="MINISBLACK")

    for ax in axes.flatten():
        ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.savefig(output_path)

    plt.show()


if __name__ == "__main__":
    level = 2
    stains = ["HE", "GS", "CYP2E1", "CYP1A2", "CYP3A4", "CYP2D6"]
    data_dir_registered: Path = Path(
        f"/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual_registered_L{level}")

    results_dir = data_dir_registered / "__results__"

    subject_dirs = sorted([f for f in data_dir_registered.iterdir() if f.is_dir() if not f.name.startswith("__")])
    for subject_dir in subject_dirs:
        print("-" * 80)
        print(subject_dir)
        print("-" * 80)

        # create_registration_image(subject_dir=subject_dir, stains=stains, results_dir=results_dir)
        create_stain_separation_image(subject_dir=subject_dir, stains=stains, results_dir=results_dir)

"""
- NOR-026, CYP2D6 staining has a large hole, repeat if possible
- UKJ-19-041, CYP3A4, slightly lighter area (analysis possible)
- UKJ-19-049, CYP3A4, slightly lighter area (analysis possible)
- MNT-021, CYP1A2, staining problem upper part

- SSES2021 12, CYP1A2, slightly lighter area (analysis possible)

- NOR-025, HE stain not registered correctly (algorithm issue) -> fix by only doing liver
- SSES2021 14, registration not working (algorithm issue)
"""
