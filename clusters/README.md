# Code for performing DP-Means Clustering via Cosine Similarity Metric

This code is a retrofitted version of DP-Means clustering as released by Dineri et al. https://github.com/BGU-CS-VIL/pdc-dp-means/tree/main/paper_code

###  MiniBatch PDC-DP-Means via Cosine Similarity

In order to install this, you must clone scikit-learn from: `https://github.com/scikit-learn/scikit-learn.git`.

Navigate to the directory `sklearn/cluster` and replace the files `__init__.py`, `_k_means_lloyd.pyx` and `_kmeans.py` with the respective files under the `cluster` directory.
Next, you need to install sklearn from source. To do so, follow the directions here: https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge.

Now, in order to use it, you can simply use `from sklearn.cluster import MiniBatchDPMeans, DPMeans`. In general, the parameters are the same as the `K-Means` counterpart:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
