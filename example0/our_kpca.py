from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# How to compute matrix K ?

class kPCA():
    l = 0  # Number of samples
    N = 0  # Dimensionality

    def __init__(self, train_data, test_data):
        """
        :param train_data: samples from the train data
        :param test_data: samples from the test data
        """
        self.Xtrain = train_data
        self.Xtest = test_data
        self.l = np.size(train_data, 0)
        self.N = np.size(train_data, 1)



    def algorithm(self, n, c):
        # Obtain RBF Kernel Matrix
        self.Ktrain = self.obtain_rbf_kernel_matrix(n, c)  # l x l
        print("Kernel matrix for train obtained")
        # Obtain eigenvectors from K
        self.alphas = self.obtain_alphas(self.Ktrain, n)  # l x n
        print("Alphas obtained")
        # Obtain RBF Kernel Matrix for test data
        self.Ktest = self.obtain_test_rbf_kernel_matrix(n, c, self.Ktrain)
        print("Kernel matrix for test obtained")
        # Obtain betas
        self.betas = np.dot(self.Ktest, self.alphas)
        # Obtain gammas
        self.gammas = np.dot(self.betas, self.alphas.T)


    def obtain_alphas(self, K, n):
        """
        :param K: RBF Kernel matrix of the train data
        :param n: number of components used in reconstruction
        :return: returns the first n eigenvectors of the K matrix,
                  as a matrix of size l x n, i.e. (number of train data) x (
                  number of components).
        """
        # Obtain eigenvalues and eigenvectors of K. The eigenvalue lambda_[i]
        # corresponds to the eigenvector alpha[:,i].
        lambda_, alpha = eigh(K)
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
        # Compute squared euclidean distances between all samples, store values
        # in a matrix
        sqdist_XtX = euclidean_distances(self.Xtest, self.X)**2
        # Apply Kernel to each value
        Ktest = np.exp(-sqdist_XtX / (n * c))
        return self.center_kernel_matrix(Ktest, Ktrain)

    def obtain_rbf_kernel_matrix(self, n, c):
        """
        :param n: number of components used in reconstruction
        :param c: parameter for the RBF Kernel function
        :return: Kernel matrix, where the coefficient K_ij = phi(xi)*phi(xj)
        """
        # Compute squared euclidean distances between all samples, store values
        # in a matrix
        sqdist_X = euclidean_distances(self.Xtrain, self.Xtrain)**2
        K = np.exp(-sqdist_X/(n*c))
        return self.center_kernel_matrix(K, K)

    def center_kernel_matrix(self, K, Ktrain):
        """
        :param K: non-centered Kernel matrix
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


"""
References

[1] Schoelkopf, Bernhard, Support vector learning, 1997
"""