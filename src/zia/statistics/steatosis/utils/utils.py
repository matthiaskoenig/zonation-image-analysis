from typing import List, Tuple, Dict
def _species_colors_rgb(species_colors) -> Dict[str, Tuple[float]]:
    return {sp: (tuple(int(h.strip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))) for sp, h in species_colors.items()}

SPECIES_ORDER = ["mouse", "rat", "human"]
PROTEIN_ORDER = ["he", "gs", "cyp1a2", "cyp2d6", "cyp2e1", "cyp3a4"]
SPECIES_COLORS = {"mouse": "#77AADD", "rat": "#EE8866", "human": "#44BB99"}
SPECIES_COLORS_RGB = _species_colors_rgb(SPECIES_COLORS)
PIXEL_SIZE = 0.2272 # µm


def map_to_group(species, diet):
    if species in ["mouse", "rat"]:
        return f"{species} ({diet}W HDF)"
    return species

def create_data_dict(attributes, data_frame):
    species_gb = data_frame.groupby("species")
    species_dict = {}  # dict of species, groups, attributes
    for i, sp in enumerate(SPECIES_ORDER):
        sp_df = species_gb.get_group(sp)
        group_dict = {}
        for gr, gr_df in sp_df.groupby("group"):
            attr_dict = {}
            for attr in attributes:
                attr_dict[attr] = gr_df[attr]
            group_dict[gr] = attr_dict
        species_dict[sp] = group_dict
    return species_dict