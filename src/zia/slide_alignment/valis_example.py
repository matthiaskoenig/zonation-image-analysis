"""Image registration with VALIS.

sudo apt-get install libvips
sudo apt-get install maven

docker run --memory=40g  -v "$HOME:$HOME" -v "/media/mkoenig/Extreme Pro/data/cyp_species_comparison:/media/mkoenig/Extreme Pro/data/cyp_species_comparison" cdgatenbee/valis-wsi python3 /home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/valis_example.py

Traceback (most recent call last):
  File "/home/mkoenig/git/zonation-image-analysis/src/zia/slide_alignment/valis_example.py", line 30, in <module>
    registrar.warp_and_save_slides(
  File "/usr/local/src/valis/valtils.py", line 29, in wrapper
    return f(*args, **kwargs)
  File "/usr/local/src/valis/registration.py", line 4332, in warp_and_save_slides
    slide_obj.warp_and_save_slide(dst_f=dst_f, level = level,
  File "/usr/local/src/valis/valtils.py", line 29, in wrapper
    return f(*args, **kwargs)
  File "/usr/local/src/valis/registration.py", line 961, in warp_and_save_slide
    warped_slide = self.warp_slide(level=level, non_rigid=non_rigid,
  File "/usr/local/src/valis/valtils.py", line 29, in wrapper
    return f(*args, **kwargs)
  File "/usr/local/src/valis/registration.py", line 892, in warp_slide
    warped_slide = slide_tools.warp_slide(src_f, M=self.M,
  File "/usr/local/src/valis/slide_tools.py", line 288, in warp_slide
    vips_warped = warp_tools.warp_img(img=vips_slide, M=M, bk_dxdy=dxdy,
  File "/usr/local/src/valis/warp_tools.py", line 1095, in warp_img
    warped = warped.extract_area(*bbox_xywh)
  File "/usr/local/src/.venv/lib/python3.10/site-packages/pyvips/vimage.py", line 1347, in call_function
    return pyvips.Operation.call(name, self, *args, **kwargs)
  File "/usr/local/src/.venv/lib/python3.10/site-packages/pyvips/voperation.py", line 305, in call
    raise Error('unable to call {0}'.format(operation_name))
pyvips.error.Error: unable to call extract_area
  extract_area: bad extract area


"""

from pathlib import Path
from valis import registration

data_dir: Path = Path("/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual")
data_dir_registered: Path = Path("/media/mkoenig/Extreme Pro/data/cyp_species_comparison/control_individual_registered")

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
        level=3,
    )

# Kill the JVM
registration.kill_jvm()

