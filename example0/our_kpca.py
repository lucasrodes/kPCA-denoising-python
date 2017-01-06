# -*- coding: utf-8 -*-
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.spatial.distance import cdist

# How to compute matrix K ?

class kPCA():
    l = 0  # Number of samples in training
    N = 0  # Dimensionality

    def __init__(self, train_data, test_data):
        """
        :param train_data: samples from the train data
        :param test_data: samples from the test data
        """
        self.Xtrain = train_data  # l_train x N
        self.Xtest = test_data  # l_test x N
        self.l_train = np.size(train_data, 0)
        self.l_test = np.size(test_data, 0)
        self.N = np.size(train_data, 1)

    def obtain_preimages(self, n, c):
        """
        :param n: number of components used in reconstruction
        :param c: parameter for the RBF Kernel function
        :return: Pre-images from all test data Xtest
        """
        # Obtain RBF Kernel Matrix
        self.Ktrain = self.obtain_rbf_kernel_matrix(n, c)  # l_train x l_train
        print("--- Kernel matrix for train obtained")
        # Obtain eigenvectors from K
        self.alphas = self.obtain_alphas(self.Ktrain, n)  # l_train x n
        print("--- Alphas obtained")
        # Obtain RBF Kernel Matrix for test data, dim: l_test x l_train
        self.Ktest = self.obtain_test_rbf_kernel_matrix(n, c, self.Ktrain)
        print("--- Kernel matrix for test obtained")
        # Obtain betas
        self.betas = np.dot(self.Ktest, self.alphas)  # l_test x n
        print("--- Betas obtained")
        # Obtain gammas, gamma[j, i] corresponds to tj (test) abd xi (train)
        self.gammas = np.dot(self.betas, self.alphas.T)  # l_test x l_train
        print("--- Gammas obtained")
        print("--- Iterative scheme started...")
        self.Z = []
        # Obtain pre-image for each test sample
        for i in range(np.size(self.Xtest, 0)):
            # Find z, pre-image
            z = self.obtain_preimage(i, n, c)
            self.Z.append(z)
            #print("---", i/363)
        self.Z = np.array(self.Z)
        print("--- Succesfully finished!")

    def obtain_preimage(self, j, n, c):
        """
        :param j: index of the test data, that is Xtest[j,:]
        :param n: number of components used in reconstruction
        :param c: parameter for the RBF Kernel function
        :return: pre-image of Xtest[j,:], that is z that minimizes |rho(z) -
                    Pn·rho(x)|^2, using eq. (10) from the paper.
        """
        z_new = self.Xtest[j, :]
        z = np.zeros((10, 1))
        n_iter = 0
        # Convergence defined as difference
        #while (np.max(z - z_new) > 0.00001) and (n_iter < 1e3):
        while (np.linalg.norm(z - z_new) > 0.0001) and (n_iter < 1e3):
            z = z_new
            zcoeff = cdist([z], self.Xtrain, 'sqeuclidean')
            zcoeff = np.exp(-zcoeff/(n*c))
            zcoeff = self.gammas[j, :] * zcoeff
            s = np.sum(zcoeff)
            zcoeff = np.sum(self.Xtrain*zcoeff.T, axis=0)
            # Avoid underflow
            if s == 0:
                s += 1e-8
            z_new = zcoeff/s
            n_iter += 1
        return z_new

    def obtain_rbf_kernel_matrix(self, n, c):
        """
        :param n: number of components used in reconstruction
        :param c: parameter for the RBF Kernel function
        :return: Kernel matrix from the train data, where the coefficient
                  K_ij = phi(xi)*phi(xj)
        """
        # Compute squared euclidean distances between all samples, store values
        # in a matrix
        sqdist_X = euclidean_distances(self.Xtrain, self.Xtrain) ** 2
        K = np.exp(-sqdist_X / (n * c))
        return self.center_kernel_matrix(K, K)

    @staticmethod
    def center_kernel_matrix(K, Ktrain):
        """
        :param K: Kernel matrix that we aim to center
        :param Ktrain: training Kernel matrix
        :return: centered Kernel matrix
        Code inspired in Appendix D.2.2 Centering in Feature Space from [1]
        """
        one_l_prime = np.ones((np.size(K, 0), np.size(K, 1))) / np.size(K, 1)
        one_l = np.ones((np.size(Ktrain, 0), np.size(Ktrain, 1))) / np.size(
            Ktrain, 1)
        K = K - one_l_prime.dot(Ktrain) - K.dot(one_l) + one_l_prime.dot(
            Ktrain).dot(one_l)
        return K

    def obtain_alphas(self, Ktrain, n):
        """
        :param K: RBF Kernel matrix of the train data
        :param n: number of components used in reconstruction
        :return: returns the first n eigenvectors of the K matrix,
                  as a matrix of size l x n, i.e. (number of train data) x (
                  number of components).
        """
        # Obtain eigenvalues and eigenvectors of K. The eigenvalue lambda_[i]
        # corresponds to the eigenvector alpha[:,i].
        lambda_, alpha = eigh(Ktrain)
        # Keep only first n eigenvalues and eigenvectors, ensure that
        # lambda_[i] (np.dot(alpha[:,i], alpha[:,i])) = 1
        lambda_n = lambda_[:-n-1:-1]
        alpha_n = np.column_stack(
            (alpha[:, -i]/np.sqrt(lambda_n[i-1]) for i in range(1, n + 1)))
        """ debugging purposes
        i_sort = np.argsort(lambda_)
        lambda_check = lambda_[i_sort[-4]]
        alpha_check = alpha[:, i_sort[-4]]
        """
        return alpha_n#, lambda_n#, lambda_check, alpha_check

    def obtain_test_rbf_kernel_matrix(self, n, c, Ktrain):
        """
        :param n: number of components used in reconstruction
        :param c: parameter for the RBF Kernel function
        :param Ktrain: Kernel matrix obtained from the train data
        :return: centered Kernel matrix crossing the data from test and train
        """
        # Compute squared euclidean distances between all samples, store values
        # in a matrix
        sqdist_XtX = euclidean_distances(self.Xtest, self.Xtrain)**2
        # Apply Kernel to each value
        Ktest = np.exp(-sqdist_XtX / (n * c))
        return self.center_kernel_matrix(Ktest, Ktrain)

"""
References

[1] Schoelkopf, Bernhard, Support vector learning, 1997
"""