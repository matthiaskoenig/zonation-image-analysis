import re
from pathlib import Path


def get_subject_from_path(image_id: str, species: str) -> str:
    """Metadata for image"""
    rat_pattern = re.compile(r"(NOR|FLR)-\d+")
    pig_pattern = re.compile(r"SSES2021 \d+")
    mouse_pattern = re.compile(r"MNT-\d+")
    human_pattern = re.compile(r"UKJ-\d{2}-\d{3}_Human")

    match = None
    if species == "pig":
        match = re.search(pig_pattern, image_id)
    elif species == "mouse":
        match = re.search(mouse_pattern, image_id)
    elif species == "rat":
        match = re.search(rat_pattern, image_id)
    elif species == "human":
        match = re.search(human_pattern, image_id)

    if match == None:
        raise Exception(f"No matching subject found for this {image_id}.")

    return match.string
