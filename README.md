# Zonation analysis

Analysis of zonation patterns from fluorscence images.

## Read images
- [ ] Read all images with bioformats
- [ ] Assign channels
- [ ] Store data

## Extract channels

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