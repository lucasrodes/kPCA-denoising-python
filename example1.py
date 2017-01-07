from sklearn.decomposition import PCA
from princurves import fit_curve
import matplotlib.pyplot as plt
import numpy as np
import our_kpca
import data

# Curves
X, Y = data.get_curves(noise='normal', scale=0.2)
noisy_data = np.array([X, Y]).T

kpca = our_kpca.kPCA(noisy_data, noisy_data)
denoised = kpca.obtain_preimages(4, 0.6)
plt.hold(True)
plt.subplot(1, 3, 1)
plt.plot(X, Y, 'k.')
plt.title('Kernel PCA')
plt.plot(denoised[:,0], denoised[:,1], 'r.')

denoised = fit_curve(noisy_data)
plt.subplot(1, 3, 2)
plt.plot(X, Y, 'k.')
plt.title('Principal Curves')
plt.plot(denoised[:,0], denoised[:,1], 'r.')

pca = PCA(n_components=1)
new_representation = pca.fit_transform(noisy_data)
denoised = pca.inverse_transform(new_representation)
plt.subplot(1, 3, 3)
plt.plot(X, Y, 'k.')
plt.title('Linear PCA')
plt.plot(denoised[:,0], denoised[:,1], 'r.')
plt.show()

# Square
X, Y = data.get_square()
noisy_data = np.array([X, Y]).T

kpca = our_kpca.kPCA(noisy_data, noisy_data)
denoised = kpca.obtain_preimages(4, 0.6)
plt.hold(True)
plt.subplot(1, 4, 1)
plt.plot(X, Y, 'k.')
plt.title('Kernel PCA')
plt.plot(denoised[:,0], denoised[:,1], 'r.')

denoised = fit_curve(noisy_data)
plt.subplot(1, 4, 2)
plt.plot(X, Y, 'k.')
plt.title('Principal Curves')
plt.plot(denoised[:,0], denoised[:,1], 'r.')

denoised = fit_curve(noisy_data, circle=True)
plt.subplot(1, 4, 3)
plt.plot(X, Y, 'k.')
plt.title('Principal Curves (starting from a circle)')
plt.plot(denoised[:,0], denoised[:,1], 'r.')

pca = PCA(n_components=1)
new_representation = pca.fit_transform(noisy_data)
denoised = pca.inverse_transform(new_representation)
plt.subplot(1, 4, 4)
plt.plot(X, Y, 'k.')
plt.title('Linear PCA')
plt.plot(denoised[:,0], denoised[:,1], 'r.')
plt.show()
