# Learning-to-Index-for-Nearest-Neighbor-Search
Chih-Yi Chiu, Member, IEEE, Amorntip Prayoonwong, and Yin-Chih Liao

# Abstract
In this study, we present a novel ranking model based on learning the nearest neighbor relationships embedded in the index
space. Given a query point, a conventional nearest neighbor search approach calculates the distances to the cluster centroids, before
ranking the clusters from near to far based on the distances. The data indexed in the top-ranked clusters are retrieved and treated as
the nearest neighbor candidates for the query. However, the loss of quantization between the data and cluster centroids will inevitably
harm the search accuracy. To address this problem, the proposed model ranks clusters based on their nearest neighbor probabilities
rather than the query-centroid distances to the query. The nearest neighbor probabilities are estimated by employing neural networks
to characterize the neighborhood relationships as a nonlinear function, i.e., the density distribution of nearest neighbors with respect to the query. The proposed probability-based ranking model can replace the conventional distance-based ranking model as a coarse filter
for candidate clusters, and the nearest neighbor probability can be used to determine the data quantity to be retrieved from the
candidate cluster. Our experimental results demonstrated that implementation of the proposed ranking model for two state-of-the-art
nearest neighbor quantization and search methods could boost the search performance effectively in billion-scale datasets.

Index Terms—cluster reranking, hash-based indexing, nearest neighbor distribution probability, optimized product quantization,
residual vector quantization.

# Full Paper
  Our full paper is now available online.
  
  View full paper: https://arxiv.org/pdf/1807.02962.pdf

# The 1000 NNs in the ground truth of DEEP1B
  The 1000 NNs in the ground truth of DEEP1B is available to download from: 
  
  https://drive.google.com/file/d/1YGWwY_q968IIStC8cdrjR5p2QxZxw-5r/view?usp=sharing
  
# Source Code
  Our source codes and data for SIFT1B RVQ 4096x4096 is available at:
  
  https://drive.google.com/drive/folders/13N6jJGB8p1yDCpa-a1JRq36SM9ZbfNdj?usp=sharing
  
  These codes are in the coarse search process. We simulate how to retrieve the candidate second-level clusters for SIFT1B RVQ before performing the ADC. Then, we calculate the top-1 recall along with the candidate second-level clusters.
  
  Only source codes are also provided here :
  The main procedure :
  https://github.com/AmorntipPrayoonwong/Learning-to-Index-for-Nearest-Neighbor-Search/blob/master/main_experiment.py
  
  The sub procedures are in the library folder lib :
  https://github.com/AmorntipPrayoonwong/Learning-to-Index-for-Nearest-Neighbor-Search/tree/master/lib
  
  Preprocessing & Learning code examples can be downloaded from: https://drive.google.com/drive/folders/1w9IffP6jeUOZMCF9pm3ybQDRRz2vwPqD?usp=sharing
  
  IVF file is available to download from: https://drive.google.com/drive/folders/15Xuk9jEEcZdD0x0ECz2PuCoZuw0RHoQf?usp=sharing


