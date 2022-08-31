# Zonation analysis

Analysis of zonation patterns from fluorscence images. Double stainings with CYP2E1 and E-Cadharin were used to determine the zonation pattern of the liver

## Normalization of channels
- [ ] updated histogram normalization. In the first version a simple quantile cutoff is used. Better approaches are to determine the background values and which to remove.
- how should the removed data be handled (mean filter?, NaNs)

## Gauss filtering
- [ ] adapt the gauss filter according to the spatial resolution of the system. I.e. the resolution parameter of the system must be used to adapt the filter size.

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