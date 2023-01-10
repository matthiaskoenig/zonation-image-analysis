"""
Work with IHC whole-slide scans.

https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html

H&E staining
Hematoxylin (stains nuclei) and eosin (everything else)


122 - Normalizing H&E images and digitally separating Hematoxylin and Eosin components
https://www.youtube.com/watch?v=yUrwEYgZUsA


1. read WSI (HE & CYP1A2)
2. separate stains
3. normalization of channels
4. zonation pattern (CYP2E1)
"""
