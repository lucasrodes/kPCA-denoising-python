from sklearn.decomposition import PCA
from our_kpca import kPCA
from princurves import fit_curve
import matplotlib.pyplot as plt
import numpy as np
import our_kpca
import data

def plot(methods, X, Y, line):
    "Plots all results in the input list as a series of subplots"
    n_methods = len(methods)
    i = 1
    plt.hold(True)
    handles = []
    for denoised, name in methods:
        plt.subplot(2, 4, i + 4*line)
        handle1, = plt.plot(X, Y, 'k.')
        plt.title(name)
        handle2, = plt.plot(denoised[:,0], denoised[:,1], 'r.')
        i += 1
        handles.append(handle1)
        handles.append(handle2)
    return handles

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
plot(methods, X, Y, 0)

# Square
X, Y = data.get_square(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T
methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.6), 'Kernel PCA'),
    (fit_curve(noisy_data), 'Principal Curves'),
    (fit_curve(noisy_data, circle=True), 'Principal Curves (from circle)'),
    (pca_denoising(noisy_data), 'Linear PCA')
]
handles = plot(methods, X, Y, 1)
plt.figlegend(handles[:2], ['Original data', 'Denoised data'],
    loc='upper right', bbox_to_anchor=(0.95, 0.75))
plt.show()
