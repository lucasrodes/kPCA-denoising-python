from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import our_kpca
import data

for shape in (data.get_curves, data.get_square):
    X, Y = shape(noise='normal', scale=0.2)
    noisy_data = np.array([X, Y]).T

    kpca = our_kpca.kPCA(noisy_data, noisy_data)
    denoised = kpca.obtain_preimages(4, 0.6)
    plt.hold(True)
    plt.subplot(1, 2, 1)
    plt.plot(X, Y, 'k.')
    plt.title('Kernel PCA')
    plt.plot(denoised[:,0], denoised[:,1], 'r.')

    pca = PCA(n_components=1)
    new_representation = pca.fit_transform(noisy_data)
    denoised = pca.inverse_transform(new_representation)
    plt.subplot(1, 2, 2)
    plt.plot(X, Y, 'k.')
    plt.title('Linear PCA')
    plt.plot(denoised[:,0], denoised[:,1], 'r.')
    plt.show()
