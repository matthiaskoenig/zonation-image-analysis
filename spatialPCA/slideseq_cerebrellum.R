# https://lulushang.org/SpatialPCA_Tutorial/slideseq.html


rm(list = ls())
setwd("/home/mkoenig/git/zonation-image-analysis/spatialPCA")

load("./data/slideseq.rds") 
print(dim(sp_count)) # The count matrix:  (proteins, locations)
print(dim(location)) # The location matrix: (x,y), locations


