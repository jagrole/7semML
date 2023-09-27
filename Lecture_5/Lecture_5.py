# ###################################
# Group ID : 985
# Members : Jakob G. Olesen, Mads Lindeborg Larsen, Sif Bjerre Lindby
# Date : September 27 2023
# Lecture: 5 clustering
# Dependencies: numpy, scipy.stats, scipy.io, matplotlib.pyplot, sklearn.decomposition, sklearn.decomposition, sklearn.mixture
# Python version: 3.11.3
# Functionality: This script computes a GMM model as compares the covariances and means to similar data for a PCA model
# ###################################


from scipy.io import loadmat
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA


# # Exercise 5: Clustering
# This assignment is based on the previously generated 2-dimensional data of the three classes (5, 6 and 8) from the MNIST database of handwritten digits. 
# 
# First, mix the 2-dimensional data (training data only) by removing the labels and then use one Gaussian mixture model to model them. 
# 
# Secondly, compare the Gaussian mixture model with the Gaussian models trained in the previous assignment, in terms of mean and variance values as well as through visualisation.


# ## Loading the data and mixing
# First we load the exercise data set, combine the individual training sets into one and shuffle the data to ensure a random shuffle (here with a seed to ensure reproducability). 



data_path = "2D568class.mat"
data = loadmat(data_path)
train5 = data["trn5_2dim"]/255
train6 = data["trn6_2dim"]/255
train8 = data["trn8_2dim"]/255

trainset = np.concatenate([train5, train6, train8])
np.random.seed(0)
np.random.shuffle(trainset)


# Gaussian Mixture model

GMM_data = GMM(n_components=3).fit(trainset)
GMM_means = GMM_data.means_
GMM_cov = GMM_data.covariances_

mean1_gmm = GMM_means[0]
mean2_gmm = GMM_means[1]
mean3_gmm = GMM_means[2]

cov1_gmm = GMM_cov[0]
cov2_gmm = GMM_cov[1]
cov3_gmm = GMM_cov[2]

# Gaussian models using PCA
decomp = PCA(n_components=2)
decomp.fit(trainset)

x5PCA = decomp.transform(train5)
x6PCA = decomp.transform(train6)
x8PCA = decomp.transform(train8)

#PCA mean, variance and Covariance matrices
mean5 = np.mean(x5PCA, axis= 0)
cov5 = np.cov(x5PCA.T)

mean6 = np.mean(x6PCA, axis= 0)
cov6 = np.cov(x6PCA.T)

mean8 = np.mean(x8PCA, axis= 0)
cov8 = np.cov(x8PCA.T)


 # ## Comparing means and covariance matrices.
# Let's look at the means and covariance matrices.
# First we extract the means and covariances from the GMM.


# Now we can compare the GMM means and covariances to the Gaussin models estimated for each class individually.


# ### Means



for name, mean in {"mean5": mean5, "mean6": mean6, "mean8": mean8, 
                   "mean1_gmm": mean1_gmm, "mean2_gmm": mean2_gmm, "mean3_gmm": mean3_gmm}.items():
    print(f"{name}: {np.array2string(mean)}")


### Covariances


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].matshow(cov5)
for (i, j), z in np.ndenumerate(cov5):
    axs[0, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 0].set_title("Cov. Class: 5")

axs[1, 0].matshow(cov1_gmm)
for (i, j), z in np.ndenumerate(cov1_gmm):
    axs[1, 0].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 0].set_title("Cov. GMM kernel 1")

axs[0, 1].matshow(cov6)
for (i, j), z in np.ndenumerate(cov6):
    axs[0, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 1].set_title("Cov. Class: 6")

axs[1, 1].matshow(cov2_gmm)
for (i, j), z in np.ndenumerate(cov2_gmm):
    axs[1, 1].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 1].set_title("Cov. GMM kernel 2")

axs[0, 2].matshow(cov8)
for (i, j), z in np.ndenumerate(cov8):
    axs[0, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[0, 2].set_title("Cov. Class: 8")

c = axs[1, 2].matshow(cov1_gmm)
for (i, j), z in np.ndenumerate(cov3_gmm):
    axs[1, 2].text(j, i, f'{z:0.1f}', ha='center', va='center')
axs[1, 2].set_title("Cov. GMM kernel 3")

plt.show()
# What do we see when comparing means and covariances?

# ## Visualizing the models in contourplots.
# Now we would like to visualize our models to compare them.


# We first generate some points to be able to sample from the models.



# Now we sample from the models using the generated points.


# The model samples can then be visualized in a contour plot.


#Plot contours for the GMM, seperated GMM and individual estimated densities



