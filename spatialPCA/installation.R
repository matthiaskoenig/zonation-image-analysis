# Installation of spatialPCA
# see information on https://lulushang.org/SpatialPCA_Tutorial/Installation.html

# sudo apt-get install libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libcurl4-openssl-dev


# Install devtools, if necessary
if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")
# Install SpatialPCA
devtools::install_github("shangll123/SpatialPCA")
# load SpatialPCA
library(SpatialPCA)
