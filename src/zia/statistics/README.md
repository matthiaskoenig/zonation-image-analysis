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
| log         | if True, analysis was done on log transformed data |
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
```overview-for-already``` contains Overview plots for containing the


