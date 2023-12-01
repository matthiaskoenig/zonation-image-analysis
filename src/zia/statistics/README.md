# Supplementary data

This directory provides the data that was used to generate the figures
and analyses presented in the paper.

# Descriptive statistics

```descriptive-stats/descriptive-stats.xlsx``` contains the descriptive statistics data that was used to generate the boxplots for the statistics of
lobule gemoetry.
The file contains sheets for species-comparison, subject-comparison for each species and the lobe comparison for each mouse subject, respectively.
The following list specifies the column name and description.

| Column      | Description                                        |
|-------------|----------------------------------------------------|
| nominal_var | column by which dataframe was grouped              |
| group       | group name                                         |
| attr        | variable of statistical analysis                   |
| unit        | variable unit                                      |
| mean        | mean                                               |
| std         | standard deviation                                 |
| se          | standard error of the mean                         |
| median      | median                                             |
| min         | minimum                                            |
| max         | maximum                                            |
| q1          | 1st quartile (25%)                                 |
| q3          | 3rd quartile (75%)                                 |
| n           | number of sample                                   |

# Expression gradient

```distance-data/lobule_distances.csv``` contains the portality measure and the normalized intensity for each pixel in a lobule and for all the
protein slides. In the following table, the column names and descriptions are prvided
| lobule |width,height,d_portal,d_central,pv_dist,intensity,protein,roi,subject,species | column by which dataframe was grouped |
The pixel position is given for resolution level seven.

| Column    | Description                                        |
|-----------|----------------------------------------------------|
| lobule    | lobule number                                      |
| width     | width (x-position) of pixel                        |
| height    | height (y-position) of pixel                       |
| d_portal  | portal distance                                    |                                                                                 
| d_central | central distance                                   |
| pv_dist   | portality (1 - d_central / (d_central + d_portal)) |
| intensity | normalized intensity                               |
| protein   | CYP protein                                        |
| roi       | ROI number                                         |
| subject   | subject                                            |
| species   | species                                            |

# Overview plots for segementation results and distance maps

```overview-for-already/``` contains overview plots for all subject and ROIs across all species in the dataset.
The plots panels display from left to right:

- HE staining of the liver tissue sample
- Overlay of DAB channels resulting from stain separation of CYP3A4, CYP2E1, GS
- Boundaries of lobular regions and vessels on DAB channel of CYP2E1
- Heap map of portality (central-portal distance)

# Paper plots

```paper-plots/``` conatains the figures as PNG and SVG files that are presented in the main test and supplements.

# Statistics of lobular geometry

```slide-statistics/slide-stats.xlsx``` contains the raw statistics data of the lobule polygons obtained from segmentation.
This data frame was used to generate the descriptive statistics for each featured group.

| Column                       | Description                                                                                                                                                |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| area                         | area of lobule polygon                                                                                                                                     |
| perimeter                    | perimeter of the polygon                                                                                                                                   |
| n_central_vessel             | number of central vessels contained or intersected by the lobule polygon                                                                                   |
| n_portal_vessel              | number of portal vessels contained or intersected by the lobule polygon                                                                                    |
| portal_vessel_cross_section  | summed area of the portal vessel polygons                                                                                                                  |
| central_vessel_cross_section | summed area of the central vessel polygons                                                                                                                 |
| compactness                  | isoperimetric quotient: Ratio of the area of the polygon to the area of a circle with the same perimeter (I_p = polygon_area * 4 * np.pi / perimeter ** 2) |
| area_without_vessels         | polygon area that does not intersect with vessel area                                                                                                      |
| minimum_bounding_radius      | radius of the minimum circle that encloses the polygon                                                                                                     |
| species                      | species                                                                                                                                                    |
| subject                      | subject id                                                                                                                                                 |
| roi                          | ROI id                                                                                                                                                     |
| {var}_unit                   | unit of the variable                                                                                                                                       |



