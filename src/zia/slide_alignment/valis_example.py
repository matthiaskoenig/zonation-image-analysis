"""Image registration with VALIS.

sudo apt-get install libvips
sudo apt-get install maven
docker run --memory=40g  -v "$HOME:$HOME" -v "/media/mkoenig/Extreme Pro/data/cyp_species_comparison:/media/mkoenig/Extreme Pro/data/cyp_species_comparison" cdgatenbee/valis-wsi python3 /home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/valis_example.py


"""

from pathlib import Path
from valis import registration

level = 2
data_dir: Path = Path("/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual")
data_dir_registered: Path = Path(f"/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual_registered_L{level}")

slide_dirs = sorted([f for f in data_dir.iterdir() if f.is_dir()])
for slide_dir in slide_dirs:
    results_dir = data_dir_registered / slide_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    print("-" * 80)
    print(slide_dir)
    print("\t->", results_dir)
    print("-" * 80)

    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(str(slide_dir), str(results_dir))
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Save all registered slides as ome.tiff
    registrar.warp_and_save_slides(
        str(results_dir), crop="overlap",
        level=level,
        compression="jpeg"
    )
    registrar.warp_and_merge_slides(
        str(results_dir / "merged.ome.tiff"), crop="overlap",
        level=level,
        compression="jpeg"
    )

# Kill the JVM
registration.kill_jvm()

