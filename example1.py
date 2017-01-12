from sklearn.decomposition import PCA
from our_kpca import kPCA
from princurves import fit_curve
import matplotlib.pyplot as plt
import numpy as np
import our_kpca
import data

def show_plot(methods):
    "Plots all results in the input list as a series of subplots"
    n_methods = len(methods)
    i = 1
    plt.hold(True)
    for denoised, name in methods:
        plt.subplot(1, n_methods, i)
        plt.plot(X, Y, 'k.')
        plt.title(name)
        plt.plot(denoised[:,0], denoised[:,1], 'r.')
        i += 1
    plt.show()

def pca_denoising(data):
    "Performs linear PCA denoising using sklearn"
    pca = PCA(n_components=1)
    low_dim_representation = pca.fit_transform(data)
    return pca.inverse_transform(low_dim_representation)

# To add a new method, simply add it to both methods list
# Curves
X, Y = data.get_curves(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T
methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.5), 'Kernel PCA'),
    (fit_curve(noisy_data), 'Principal Curves'),
    (pca_denoising(noisy_data), 'Linear PCA')
]
show_plot(methods)

# Square
X, Y = data.get_square(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T
methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.6), 'Kernel PCA'),
    (fit_curve(noisy_data), 'Principal Curves'),
    (fit_curve(noisy_data, circle=True), 'Principal Curves (starting from a circle)'),
    (pca_denoising(noisy_data), 'Linear PCA')
]
show_plot(methods)
