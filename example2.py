from sklearn.decomposition import PCA
from our_kpca import kPCA
from princurves import fit_curve
import matplotlib.pyplot as plt
import numpy as np
import our_kpca
import data_example2 as data

# Enable this to use LaTeX fonts in the plot labeling
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot(methods, X, Y, line, rowspan):
    "Plots all results in the input list as a series of subplots"
    n_methods = len(methods)
    i = 0
    plt.hold(True)
    handles = []
    for denoised, name in methods:
        plt.subplot2grid((3, 4), (line, i), rowspan=rowspan)
        handle1, = plt.plot(X, Y, '.', color="0.8")
        plt.title(name)
        handle2, = plt.plot(denoised[:,0], denoised[:,1], 'k.')
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
plot(methods, X, Y, 0, 1)

# Square
X, Y = data.get_square(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T
methods = [
    (kPCA(noisy_data).obtain_preimages(4, 0.6), 'Kernel PCA'),
    (fit_curve(noisy_data), 'Principal Curves'),
    (fit_curve(noisy_data, circle=True), 'Principal Curves (from circle)'),
    (pca_denoising(noisy_data), 'Linear PCA')
]
handles = plot(methods, X, Y, 1, 2)
plt.figlegend(handles[:2], ['Original data', 'Denoised data'],
    loc='upper right', bbox_to_anchor=(0.92, 0.85))
plt.show()
