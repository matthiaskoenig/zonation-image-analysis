# Installation of spatialPCA
# see information on https://lulushang.org/SpatialPCA_Tutorial/Installation.html

# sudo apt-get install libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libcurl4-openssl-dev


# Install devtools, if necessary
if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")
# Install SpatialPCA
devtools::install_github("shangll123/SpatialPCA")
install.packages("peakRAM")
install.packages("Matrix")

# python to R conversion
install.packages("RcppCNPy")
install.packages("feather")
install.packages("arrow")


# Wraps common clustering algorithms in an easily extended S4 framework. 
# Backends are implemented for hierarchical, k-means and graph-based clustering. 
# Several utilities are also provided to compare and evaluate clustering results.
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("bluster")

# load SpatialPCA
library(SpatialPCA)
library(peakRAM)
