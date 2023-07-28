# Information and Resources

Collection of links and resources.

## Whole slide images
Information on processing WSI with python is available from 

### videos
**266 openslide - library for loading whole-slide images**  
https://www.youtube.com/watch?v=QntLBvUZR5c  
creating tiles and tile structures

**267 Processing whole-slide images as tiles**  
https://www.youtube.com/watch?v=tNfcvgPKgyU&list=PLZsOBAyNTZwYx-7GylDo3LSYpSompzsqW&index=122

**122 HE stains**  
https://www.youtube.com/watch?v=yUrwEYgZUsA

**Troubleshotting HE slides**  
https://www.youtube.com/watch?v=2zufQ5VdYP8

**DeepZoom for WSI**  
https://www.youtube.com/watch?v=W3-bYFBfJT4

**How to solve whole-slide image storage?**  
https://www.youtube.com/watch?v=2zF1DWiIIVg

### pyvips
libvips is a demand-driven, horizontally threaded image processing library. Compared to similar libraries, libvips runs quickly and uses little memory. libvips is licensed under the LGPL 2.1+.

It has around 300 operations covering arithmetic, histograms, convolution, morphological operations, frequency filtering, colour, resampling, statistics and others. It supports a large range of numeric formats, from 8-bit int to 128-bit complex. Images can have any number of bands. It supports a good range of image formats, including JPEG, TIFF, PNG, WebP, FITS, Matlab, OpenEXR, PDF, SVG, HDR, PPM, CSV, GIF, Analyze, NIfTI, DeepZoom, and OpenSlide. It can also load images via ImageMagick or GraphicsMagick, letting it load formats like DICOM.

Library for putting back into whole-slide formats
```
import pyvips
```
https://libvips.github.io/pyvips/  
https://www.libvips.org/  

build pyramid image  
https://github.com/libvips/pyvips/issues/157  
https://github.com/libvips/pyvips/issues/170  

## Python libraries for image import/export
### tifffile
https://github.com/cgohlke/tifffile

Tifffile is a Python library to
* store NumPy arrays in TIFF (Tagged Image File Format) files, and
* read image and metadata from TIFF-like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, SGI, NIHImage, ImageJ, MicroManager, FluoView, ScanImage, SEQ, GEL, SVS, SCN, SIS, BIF, ZIF (Zoomable Image File Format), QPTIFF (QPI), NDPI, and GeoTIFF files.
Image data can be read as NumPy arrays or Zarr arrays/groups from strips, tiles, pages (IFDs), SubIFDs, higher order series, and pyramidal levels.
Image data can be written to TIFF, BigTIFF, OME-TIFF, and ImageJ hyperstack compatible files in multi-page, volumetric, pyramidal, memory-mappable, tiled, predicted, or compressed form.

Tifffile can also be used to inspect TIFF structures, read image data from multi-dimensional file sequences, write fsspec ReferenceFileSystem for TIFF files and image file sequences, patch TIFF tag values, and parse many proprietary metadata formats.

### AICSImageIO
Image Reading, Metadata Conversion, and Image Writing for Microscopy Images in Pure Python.
https://allencellmodeling.github.io/aicsimageio/

## Storage formats
### Zarr
https://zarr.readthedocs.io/en/stable/

Zarr is a format for the storage of chunked, compressed, N-dimensional arrays inspired by HDF5, h5py and bcolz.
* Create N-dimensional arrays with any NumPy dtype.
* Chunk arrays along any dimension.
* Compress and/or filter chunks using any NumCodecs codec.
* Store arrays in memory, on disk, inside a Zip file, on S3, …
* Read an array concurrently from multiple threads or processes.
* Write to an array concurrently from multiple threads or processes.
* Organize arrays into hierarchies via groups.




## Whole-slide image viewers
### QuPath
https://qupath.github.io/

QuPath is often used for digital pathology applications because it offers a powerful set of tools for working with whole slide images - but it can be applied to lots of other kinds of image as well.

Features include:
- Powerful annotation & visualization tools using a modern JavaFX interface
- Built-in algorithms for common tasks, including cell and tissue detection
- Interactive machine learning, both for object and pixel classification
- Compatibility with other popular open tools, including ImageJ, OpenCV, Java Topology Suite and OMERO
- Support for many image formats through Bio-Formats and OpenSlide, including whole slide images and multiplexed data
- Groovy scripting for customization and deeper data interrogation

### Napari
napari is a fast, interactive, multi-dimensional image viewer for Python. It’s designed for browsing, annotating, and analyzing large multi-dimensional images. It’s built on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the scientific Python stack (numpy, scipy). It includes critical viewer features out-of-the-box, such as support for large multi-dimensional data, and layering and annotation. By integrating closely with the Python ecosystem, napari can be easily coupled to leading machine learning and image analysis tools (e.g. scikit-image, scikit-learn, TensorFlow, PyTorch), enabling more user-friendly automated analysis.

https://napari.org/stable/
https://www.youtube.com/watch?v=vismuuc4y1I&list=PLYx7XA2nY5Gfxu98P_HL1MnFb_BSkpxLV&index=11

### Openseadragon & DeepZoom
https://openseadragon.github.io/

An open-source, web-based viewer for high-resolution zoomable images, implemented in pure JavaScript, for desktop and mobile.

Overlay images:  
https://github.com/openseadragon/openseadragon

https://github.com/openslide/openslide-python/blob/main/examples/deepzoom/deepzoom_server.py  
https://github.com/openseadragon/openseadragon/issues/1399  
https://github.com/WasMachenSachen/openseadragon-opacity-slider  


## Image augmentation
Often necessary to augment images
https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html

One of the best ways to reduce overfitting is to collect more (good-quality) data. However, collecting more data is not always feasible or can be very expensive. A related technique is data augmentation.

Data augmentation involves generating new data records or features from existing data, expanding the dataset without collecting more data. It helps improve model generalization by creating variations of original input data and making it harder to memorize irrelevant information from training examples or features. Data augmentation is common for image and text data, but also exists for tabular data.
