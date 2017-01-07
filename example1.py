import matplotlib.pyplot as plt
import numpy as np
import our_kpca
import data

X, Y = data.get_curves(noise='normal', scale=0.2)
circles = np.array([X, Y]).T
kpca = our_kpca.kPCA(circles, circles)
denoised = kpca.obtain_preimages(4, 1)

plt.hold(True)
plt.plot(X, Y, 'k.')
plt.plot(denoised[:,0], denoised[:,1], 'r.')
plt.show()
