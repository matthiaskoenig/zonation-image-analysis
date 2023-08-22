"""Image registration with VALIS.

sudo apt-get install libvips
sudo apt-get install maven
docker run --memory=40g  -v "$HOME:$HOME" -v "/media/mkoenig/Extreme Pro/data/cyp_species_comparison:/media/mkoenig/Extreme Pro/data/cyp_species_comparison" cdgatenbee/valis-wsi python3 /home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/valis_example.py


"""

from pathlib import Path
import os

from zia import BASE_PATH
from zia.config import read_config

config = read_config(BASE_PATH / "configuration.ini")
vipsbin = str(config.libvips_path)
# print(vipsbin)

os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
os.add_dll_directory(vipsbin)
for s in os.environ["PATH"].split(";"):
    print(s)
from valis import registration

level = 0
data_dir: Path = Path(r"D:\image_data\rois_wsi")
data_dir_registered: Path = Path(r"D:\image_data\rois_registered")

subject_dirs = sorted([f for f in data_dir.iterdir() if f.is_dir()])
for subject_dir in subject_dirs:
    slide_dirs = sorted([f for f in subject_dir.iterdir() if f.is_dir()])
    for slide_dir in slide_dirs:

        results_dir = data_dir_registered / subject_dir.name / slide_dir.name
        results_dir.mkdir(parents=True, exist_ok=True)
        if any(results_dir.iterdir()):
            print(f"{results_dir} exists and not empty... skipped")
            continue
        print("-" * 80)
        print(slide_dir)
        print("\t->", results_dir)
        print("-" * 80)

        # Create a Valis object and use it to register the slides in slide_src_dir
        registrar = registration.Valis(str(slide_dir), str(results_dir))
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        # Save all registered slides as ome.tiff
        registrar.warp_and_save_slides(
            str(results_dir),
            crop="reference",
            level=level,
            compression="jpeg",
            tile_wh=2 ** 11
        )
        # registrar.warp_and_merge_slides(
        #     str(results_dir / "merged.ome.tiff"), crop="overlap",
        #     level=level,
        #     compression="jpeg"
        # )

# Kill the JVM
registration.kill_jvm()
