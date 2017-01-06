# -*- coding: utf-8 -*-
import our_kpca
import numpy as np
from scipy.spatial.distance import cdist

N = 10  # Sample space dimension
M = 11  #  Number of Gaussians
S = [100, 33]  #  Samples selected from each source for [training, test]
Sigma = [0.05, 0.1, 0.2, 0.4, 0.8]  # Variance values

for sigma in Sigma:
    c = 2*sigma**2
    centers = np.random.uniform(low=-1.0, high=1.0, size=(M, N))  # M x N

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

    kpca = our_kpca.kPCA(train_data, test_data)

    for n in range(1, N):
        print("====================")
        print("sigma = ", sigma, "n =", n)
        print("====================")
        kpca.obtain_preimages(n, c)
        Z = kpca.Z

        mse = 0
        for i in range(np.size(test_data, 0)):
            mse += cdist([Z[i, :]], [centers[i // 33]], 'sqeuclidean')
        mse /= S[1]
        print("MSE = ", mse)
        print("")
    # alpha_n, lambda_n = kpca.algorithm(10, 1)

    # for i in range(np.size(alpha_n, 1)):
    #    print lambda_n[i]*np.dot(alpha_n[:, i], alpha_n[:, i])
