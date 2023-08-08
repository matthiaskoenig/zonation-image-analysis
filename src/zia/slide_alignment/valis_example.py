"""Image registration with VALIS.

sudo apt-get install libvips
sudo apt-get install maven

docker run --memory=60g  -v "$HOME:$HOME" cdgatenbee/valis-wsi python3 /home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/valis_example.py

"""


from valis import registration

species = "mouse"
slide_src_dir = f"/home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/slides/{species}"
results_dst_dir = f"/home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/results/{species}"
registered_slide_dst_dir = f"/home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/registered_slide/{species}"

# Create a Valis object and use it to register the slides in slide_src_dir
registrar = registration.Valis(slide_src_dir, results_dst_dir)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Save all registered slides as ome.tiff
registrar.warp_and_save_slides(
    registered_slide_dst_dir, crop="overlap",
    level=3,
)

# Kill the JVM
registration.kill_jvm()

