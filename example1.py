# -*- coding: utf-8 -*-
import our_kpca
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import pandas as pd

N = 10  # Sample space dimension
M = 11  #  Number of Gaussians
S = [100, 33]  #  Samples selected from each source for [training, test]
Sigma = [0.05, 0.1, 0.2, 0.4, 0.8]  # Std deviation values
ratio_mse = pd.DataFrame(index= Sigma, columns = range(1, N)) # Table to
# show end results
count = 1  # For printing purposes

# Iterate over the considered values of sigma
for sigma in Sigma:
    c = 2*sigma**2  # Define c for the RBF Kernel

    # GENERATE TRAIN AND TEST DATA #
    # Pick centers of the M Gaussians
    centers = np.random.uniform(low=-1.0, high=1.0, size=(M, N))  # M x N
    # Construct train_data as a matrix of dimension (S[0]*M)xN,
    # that is sample S[0] samples for each Gaussian. Each row of the
    # matrix is an N-dimensional sample
    train_data = np.random.multivariate_normal(mean=centers[0],
                                               cov=sigma ** 2 * np.eye(N),
                                               size=S[0])
    for i in range(1, M):
        train_data = np.concatenate((train_data,
                                     np.random.multivariate_normal(
                                         mean=centers[i],
                                         cov=sigma ** 2 * np.eye(N),
                                         size=S[0])), axis=0)
    # Similarly to the train_data matrix, construct the test_data as a
    # matrix of dimension (S[1]*M)xN
    test_data = np.random.multivariate_normal(mean=centers[0],
                                              cov=sigma ** 2 * np.eye(N),
                                              size=S[1])
    for i in range(1, M):
        test_data = np.concatenate((test_data,
                                    np.random.multivariate_normal(
                                        mean=centers[i],
                                        cov=sigma ** 2 * np.eye(N),
                                        size=S[1])), axis=0)

    # Training data with zero mean, subtract the mean also to test data and centers.
    # I think is necessary for PCA... but idk!
    mu = np.mean(train_data, 0)
    train_data -= mu
    test_data -= mu
    centers -= mu

    # RUN KPCA OVER THE DATA #
    # Initialize the kernel PCA object
    kpca = kPCA(train_data, test_data)

    # Iterate over all considered number of components
    for n in range(1, N):
        print("====================")
        print("sigma = ", sigma, "n =", n)
        print("====================")
        # Obtain preimages of all test samples (kPCA)
        kpca.obtain_preimages(n, c)
        Z = kpca.Z

        # Obtain low-representation of test samples using PCA
        pca = PCA(n_components=n)
        pca.fit(train_data)
        test_transL = pca.transform(test_data)
        ZL = pca.inverse_transform(test_transL)

        # Obtain MSE using kPCA and PCA
        mse_kpca = mse_pca = 0
        for i in range(np.size(test_data, 0)):
            mse_kpca += cdist([Z[i, :]], [centers[i // 33]], 'sqeuclidean')
            mse_pca += cdist([ZL[i, :]], [centers[i // 33]], 'sqeuclidean')
        mse_kpca /= S[1]
        mse_pca /= S[1]
        # Obtain the ratio
        ratio_mse.set_value(sigma,n, mse_pca[0][0]/mse_kpca[0][0])

        # Information for user
        #"""
        print("")
        print("ratio_MSE =", mse_pca[0][0]/mse_kpca[0][0])
        print("kPCA MSE = ", mse_kpca[0][0])
        print("PCA MSE = ", mse_pca[0][0])
        print("")
        print(count, "/", (len(Sigma)*(N-1)))
        print("")
        count += 1

# PRINT FINAL RESULTS
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(ratio_mse)