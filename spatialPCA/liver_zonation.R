# liver zonation
# analysis of zonation patterns with spatialPCA
# https://github.com/shangll123/SpatialPCA

# https://lulushang.org/SpatialPCA_Tutorial/slideseq.html
# https://lulushang.org/docs/Projects/SpatialPCA/Slideseq

library(RcppCNPy)
library(feather)
library(SpatialPCA)
library(peakRAM)

rm(list = ls())
setwd("/home/mkoenig/git/zonation-image-analysis/spatialPCA")

df_location <- arrow::read_feather('NOR-024_R0_L6_location.feather')
location <- mat <- as.matrix(df_location)
df_spcount <- feather::read_feather('NOR-024_R0_L6_location.feather')



location1 <- npyLoad("NOR-024_R0_L4_location.npy")
rownames(location1, c("xcoord", "ycoord"))
colnames(location1, c)


# set dimnames and positions
sp_count1 <- npyLoad("NOR-024_R0_L4_spcount.npy")




load("./data/slideseq.rds") 
print(dim(sp_count)) # The count matrix:  (proteins, locations)
print(dim(location)) # The location matrix: (x,y), locations

# -----------------
# Run spatialPCA
# -----------------
mem_sparse1 <- peakRAM({
  start_time <- Sys.time()
  slideseq = CreateSpatialPCAObject(
    counts=sp_count, 
    location=location, 
    project = "SpatialPCA", 
    gene.type="spatial",
    sparkversion="sparkx",
    numCores_spark=12, 
    customGenelist=NULL,
    min.loctions = 20, 
    min.features=20
  )
  end_time <- Sys.time()
  T_sparse1 = end_time - start_time
})
mem_sparse1
T_sparse1

# > mem_sparse1
#   Elapsed_Time_sec Total_RAM_Used_MiB Peak_RAM_Used_MiB
# 1          147.783              141.9            9963.9
# > T_sparse1
# Time difference of 2.463042 mins

mem_sparse1 <- peakRAM({
  start_time <- Sys.time()
  slideseq = SpatialPCA_buildKernel(
    slideseq, 
    kerneltype="gaussian", 
    bandwidthtype="Silverman",
    bandwidth.set.by.user=NULL,
    sparseKernel=TRUE,
    sparseKernel_tol=1e-20,
    sparseKernel_ncore=10
  )
  slideseq = SpatialPCA_EstimateLoading(slideseq, fast=TRUE, SpatialPCnum=20)
  slideseq = SpatialPCA_SpatialPCs(slideseq, fast=TRUE)
  end_time <- Sys.time()
  T_sparse1 = end_time - start_time
})
mem_sparse1
T_sparse1

## Selected kernel type is:  gaussian  
## The bandwidth is:  0.00870514399377566  
## Calculating sparse kernel matrix
#   Elapsed_Time_sec Total_RAM_Used_MiB Peak_RAM_Used_MiB
# 1         1329.051               4539           11439.8
# > T_sparse1
# Time difference of 22.15085 mins

# save results
SpatialPCA_result = list()
SpatialPCA_result$SpatialPCs  = as.matrix(slideseq@SpatialPCs)
SpatialPCA_result$normalized_expr  = slideseq@normalized_expr
SpatialPCA_result$location = slideseq@location
save(SpatialPCA_result, file = "slideseq_SpatialPCA_result.RData")


# -------------------------
# Obtain clustering result
# -------------------------
clusterlabel= louvain_clustering(clusternum=8, latent_dat=slideseq@SpatialPCs, knearest=round(sqrt(dim(slideseq@SpatialPCs)[2])))

# Visualize spatial domains detected by SpatialPCA.
cbp_spatialpca <- c("lightyellow2", "coral", "lightcyan2" ,"#66C2A5", "cornflowerblue" ,"#FFD92F" ,"#E78AC3", "skyblue1")
pdf("slideseq_SpatialPCA_cluster8.pdf", width=5, height=5)
# clusterlabel = SpatialPCA_result$clusterlabel
p = plot_cluster(
  legend="right", location=SpatialPCA_result$location, clusterlabel, 
  pointsize=1, text_size=20, title_in=paste0("SpatialPCA"),color_in=cbp_spatialpca
)
p
dev.off()
