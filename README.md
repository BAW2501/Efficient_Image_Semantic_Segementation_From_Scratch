# Semantic Image Segmentation using K-Median
This repository contains code and instructions for using K-Median for semantic image segmentation. K-Median is a clustering algorithm that can be used to group pixels in an image based on their similarity, with the goal of assigning each pixel a class label.

# Requirements
To use the code in this repository, you will need the following packages:

NumPy
Matplotlib

# Usage
The main script for performing image segmentation using K-Median is k_median.py. To use it, you will need to provide the following inputs:

image: an image as 3D NumPy array representing the input
k: the number of clusters to use for K-Median
max_iter: the maximum number of iterations to run the K-Median algorithm (optional, default=100)

here is an example of how to use the script as demonstrated in the Jupyter notebook:
```
from k_median import k_median
import numpy as np
import matplotlib.pyplot as plt
# read image
img = plt.imread('NaturalColorImages.jpg')
# flatten all but last dim
pixel_vals = img.reshape((-1, img.shape[-1]))
# apply k_median
new_pixel_vals, labels = k_median(pixel_vals, n_cluster=3)
# reshape into image
segmented_image = new_pixel_vals.reshape(img.shape)
# plot image and segmented image side by side
plt.figure(figsize=(16, 9))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(segmented_image)
plt.show()
```

# References

* "K-Median Clustering." Wikipedia. Accessed January 1, 2023. https://en.wikipedia.org/wiki/K-medians_clustering.
* "Semantic Segmentation." Wikipedia. Accessed January 1, 2023. https://en.wikipedia.org/wiki/Semantic_segmentation.