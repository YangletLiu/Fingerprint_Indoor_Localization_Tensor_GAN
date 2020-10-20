# Fingerprint_Indoor_Localization_Tensor_GAN
  
  This repository contains the Python and MATLAB codes for the paper **Real-time indoor localization for smartphones using tensor-generative adversarial nets. IEEE Transactions on Neural Networks and Learning Systems**, 2020, by X.-Y. Liu and X. Wang.

# Datasets: 

1. A 20 m × 80 m floor (located in a research institute building)

There are 21 randomly deployed APs. 

Fingerprints are sampled at grid size: 0.3m x 0.3m, resulting in a fingerprint tensor: 64 x 256 x 21

2. We also include a synthetic fingerprint dataset of size: 476 x 598 x 15, used in a previous project: **Adaptive sampling of RF fingerprints for fine-grained indoor localization. IEEE Transactions on Mobile Computing, 2016. By X.-Y. Liu, S. Aeron, V. Aggarwal, X. Wang, M.-Y. Wu.**

# MATLAB codes for kNN and Direct Neural Networks

  We used MATLAB for kNN and Direct neural networks
 
1. kNN_performance implements the kNN algorithm

2. Direct neural networks (it split the data into training set and testing set, we also include the data in the MATLAB folder).


# Python Codes for GAN (regressor and discriminator)
