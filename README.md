# Zonation analysis

Analysis of zonation patterns from fluorscence images. Double stainings with CYP2E1 and E-Cadharin were used to determine the zonation pattern of the liver

## Normalization of channels
- [ ] updated histogram normalization. In the first version a simple quantile cutoff is used. Better approaches are to determine the background values and which to remove.
- how should the removed data be handled (mean filter?, NaNs)

## Gauss filtering
- [ ] adapt the gauss filter according to the spatial resolution of the system. I.e. the resolution parameter of the system must be used to adapt the filter size.

## Normalize channels


# Whole slide images

putting back into whole-slide formats
import pyvips
https://libvips.github.io/pyvips/
https://www.libvips.org/

=> build pyramid image
https://github.com/libvips/pyvips/issues/157
https://github.com/libvips/pyvips/issues/170

libvips is a demand-driven, horizontally threaded image processing library. Compared to similar libraries, libvips runs quickly and uses little memory. libvips is licensed under the LGPL 2.1+.

It has around 300 operations covering arithmetic, histograms, convolution, morphological operations, frequency filtering, colour, resampling, statistics and others. It supports a large range of numeric formats, from 8-bit int to 128-bit complex. Images can have any number of bands. It supports a good range of image formats, including JPEG, TIFF, PNG, WebP, FITS, Matlab, OpenEXR, PDF, SVG, HDR, PPM, CSV, GIF, Analyze, NIfTI, DeepZoom, and OpenSlide. It can also load images via ImageMagick or GraphicsMagick, letting it load formats like DICOM.

266 openslide
library for loading whole-slide images 
- creating tiles and tile structures

267 Processing whole-slide images as tiles
https://www.youtube.com/watch?v=tNfcvgPKgyU&list=PLZsOBAyNTZwYx-7GylDo3LSYpSompzsqW&index=122

122 HE stains
https://www.youtube.com/watch?v=yUrwEYgZUsA


Troubleshtting HE slides
https://www.youtube.com/watch?v=2zufQ5VdYP8

DeepZoom for WSI
https://www.youtube.com/watch?v=W3-bYFBfJT4


How to solve whole-slide image storage
https://www.youtube.com/watch?v=2zF1DWiIIVg

Overlay images:
https://github.com/openseadragon/openseadragon

https://github.com/openslide/openslide-python/blob/main/examples/deepzoom/deepzoom_server.py
https://github.com/openseadragon/openseadragon/issues/1399
https://github.com/WasMachenSachen/openseadragon-opacity-slider
=======
## Zonation score
- Adjust zonation score between [0, 1] with actual meaning in the colorbar (pp=0, pv=1)
- How to handle absolute quantification between the different slides ?!

## Background & Vene segmentation/detection
- necessary to detect the background


## Analysis of patterns
- [ ] create filter for zones based on thresholds (multiple thresholds), i.e., use 10 areas in histogram and map the data on it
- [ ] determine DAPI/Zone or nucleus statistics per zone; use this as an example to match additional channel data on the zonation patterns
- determine the shortest paths between minima and maxima (gradient paths), and calculate statistics along the path lines.

## Store analysis results
- [ ] write results as images (TIFF) which can be read again with standard software such as QuPath (add additional layers for). This includes the zonation patterns, but also

## Interactive exploration
- [ ] use software/tools for the interactive exploration of channels/blending. QuPath is already very nice, but an additional transparancy option would be required

## Additional data
- [ ] glycogen scans
- [ ] Whole-slide scans Uta (Mohammed), check what else will be done in the project.

## Better measurements (Scans)
- [ ] perform test whole-slide scans
- [ ] perform example scan on the Fluorescence Microscope (ITB)
