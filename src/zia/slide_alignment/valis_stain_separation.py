from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt

from zia.annotations.pipelines.stain_separation.stain_separation_whole_image import separate_raw_image
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
    console.log(subject_id)
    console.log(image_prefixes)

    # output_path = subject_dir / f"{subject_id}_registered.png"
    output_path = results_dir / f"{subject_id}_registered.png"
    image_paths = [non_rigid_dir / f"{prefix}.png" for prefix in image_prefixes.values()]
    merge_images(paths=image_paths, output_path=output_path, direction="horizontal")
    console.log(output_path)


def create_stain_separation_images(subject_dir: Path, stains: List[str], results_dir: Path):
    subject_id: str = subject_dir.name

    ome_tiff_dir = subject_dir
    image_prefixes: Dict[str, str] = image_prefixes_from_dir(ome_tiff_dir, stains=stains)
    files = [ome_tiff_dir / f"{prefix}.ome.tiff" for prefix in image_prefixes.values()]

    console.log(subject_id)
    console.log(image_prefixes)

    def plot_matrix(data: np.ndarray, path: Path, title: str):
        """Plot matrix to file."""

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
        fig.suptitle(title)

        ax.imshow(data, vmin=0, vmax=255, cmap="binary_r")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.savefig(path)
        plt.show()

    # perform stain separation and save files for processing
    for i, file in enumerate(files):
        stain: str = stains[i]
        console.print(f"{subject_id}_{stain}")

        image_array = np.array(read_ndpi(file)[0])
        channel_0: np.ndarray
        channel_1: np.ndarray
        channel_0, channel_1 = separate_raw_image(image_array)

        # save image for reading
        channel_name_0 = f"{stain}_H"
        if stain == "HE":
            channel_name_1 = f"{stain}_E"
        else:
            channel_name_1 = f"{stain}_DOB"

        for channel, name in [
            (channel_0, channel_name_0),
            (channel_1, channel_name_1),
        ]:
            # store ndarray
            path_array = results_dir / f"{subject_id}_{name}.npy"
            console.print(path_array)
            np.save(path_array, channel)

            # store ome tiff
            path_ome = results_dir / f"{subject_id}_{name}.ome.tiff"
            imwrite(path_ome, channel, photometric="MINISBLACK")

            # store image
            path_png = results_dir / f"{subject_id}_{name}.png"
            plot_matrix(
                data=channel,
                path=path_png,
                title=f"{subject_id}_{channel_name_0}"
            )


if __name__ == "__main__":
    level = 3
    stains = ["HE", "GS", "CYP2E1", "CYP1A2", "CYP3A4", "CYP2D6"]
    data_dir_registered: Path = Path(
        f"/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual_registered_L{level}")

    results_dir = data_dir_registered / "__results__"
    results_dir.mkdir(exist_ok=True, parents=True)

    subject_dirs = sorted([f for f in data_dir_registered.iterdir() if f.is_dir() if not f.name.startswith("__")])
    for subject_dir in subject_dirs:
        print("-" * 80)
        print(subject_dir)
        print("-" * 80)

        create_registration_image(subject_dir=subject_dir, stains=stains, results_dir=results_dir)
        create_stain_separation_images(subject_dir=subject_dir, stains=stains, results_dir=results_dir)

"""
- NOR-026, CYP2D6 staining has a large hole, repeat if possible
- UKJ-19-041, CYP3A4, slightly lighter area (analysis possible)
- UKJ-19-049, CYP3A4, slightly lighter area (analysis possible)
- MNT-021, CYP1A2, staining problem upper part

- SSES2021 12, CYP1A2, slightly lighter area (analysis possible)

- NOR-025, HE stain not registered correctly (algorithm issue) -> fix by only doing liver
- SSES2021 14, registration not working (algorithm issue)
"""
