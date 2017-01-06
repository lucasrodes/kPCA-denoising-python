from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np

# How to compute matrix K ?

class kPCA():
    l = 0  # Number of samples
    N = 0  # Dimensionality

    def __init__(self, train_data):
        """
        :param train_data: samples from the training
        """
        self.X = train_data
        self.l = np.size(train_data, 0)
        self.N = np.size(train_data, 1)

    def algorithm(self, n, c):
        """
        :param n: number of components used in reconstruction
        :param c: parameter for the RBF Kernel
        :return: TODO
        """
        # Obtain RBF Kernel Matrix
        K = self.obtain_rbf_kernel_matrix(n, c)
        # Obtain eigenvalues and eigenvectors of K
        # The eigenvalue lambda_[i] corresponds to the eigenvector alpha[:,i].
        lambda_, alpha = eigh(K)
        # Keep only first n eigenvalues and eigenvectors, ensure that
        # lambda_[i] (np.dot(alpha[:,i], alpha[:,i])) = 1
        lambda_n = lambda_[:-n-1:-1]
        alpha_n = np.column_stack(
            (alpha[:, -i]/np.sqrt(lambda_n[i-1]) for i in range(1, n + 1)))
        """ debugging
        i_sort = np.argsort(lambda_)
        lambda_check = lambda_[i_sort[-4]]
        alpha_check = alpha[:, i_sort[-4]]
        """
        # Obtain test Kernel matrix
        Kt = np.eye(self.l)
        
        return alpha_n, lambda_n#, lambda_check, alpha_check

    def obtain_rbf_kernel_matrix(self, n, c):
        """
        :param n: number of components used in reconstruction
        :param c: parameter for the RBF Kernel function
        :return: Kernel matrix, where the coefficient K_ij = phi(xi)*phi(xj)
        """
        # Compute euclidean distances between all samples, store values in a
        # matrix
        dist_X = pdist(self.X, 'sqeuclidean')
        # Square the distances
        sqdist_X = squareform(dist_X)
        # Apply Kernel to each value
        K = np.exp(-sqdist_X/(n*c))
        return self.center_kernel_matrix(K)

    def center_kernel_matrix(self, K):
        """
        :param K: non-centered Kernel matrix
        :return: centered Kernel matrix
        Code inspired in [1], Appendix D.2.2
        """
        one_l = np.ones((self.l, self.l)) / self.l
        K = K - one_l.dot(K) - K.dot(one_l) + one_l.dot(K).dot(one_l)
        return K


"""
References

[1] Schoelkopf, Bernhard, Support vector learning, 1997
"""