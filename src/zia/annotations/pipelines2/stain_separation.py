import traceback
from pathlib import Path
import faulthandler

from zia.annotations.pipelines2.registered_wsi_stain_separation import separate_stains
from zia.log import get_logger

data_dir_registered: Path = Path(r"D:\image_data\rois_registered")
out_path = Path(r"D:\image_data\stain_separated")
out_path.mkdir(parents=True, exist_ok=True)
logger = get_logger(__file__)
if __name__ == "__main__":
    faulthandler.enable()
    subject_dirs = sorted([f for f in data_dir_registered.iterdir() if f.is_dir()])
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        roi_dirs = sorted([f for f in subject_dir.iterdir() if f.is_dir()])
        for roi_dir in roi_dirs:
            roi = roi_dir.name
            images = sorted([f for f in roi_dir.iterdir() if f.is_file()])
            for image in images:
                try:
                    separate_stains(image, subject, roi, out_path)
                except Exception as e:
                    traceback.print_exc()
                    continue


