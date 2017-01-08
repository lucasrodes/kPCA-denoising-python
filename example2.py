import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn.datasets import fetch_mldata

from our_kpca import kPCA

# Fetch dataset and display info
usps = fetch_mldata('usps')
print('Dataset info:', usps.DESCR)
print('Datapoints: {0}\nFeatures: {1}'.format(*usps.data.shape))

# Add a secondary dataset noisy images,
# noise is gaussian with 0 mean and std=0.5,
# not sure about the 'speckle' noise in the paper.
# Also set the class labels to the number they represent
usps.update({
    'data': - usps.data,
    'gaussian': - skimage.util.random_noise(usps.data, mode='gaussian', var=0.5 ** 2, seed=123),
    's&p': - skimage.util.random_noise(usps.data, mode='s&p', amount=0.4, seed=123),
    'target': usps.target - 1
})

for noise_model in ['gaussian', 's&p']:

    # Show the first occurrence of each number, with and without noise
    vals, locs = np.unique(usps.target, return_index=True)
    for number, index in zip(vals, locs):
        plt.subplot(5, 4, 2 * number + 1)
        image = np.array(usps.data[index]).reshape((16, 16))
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title('Idx: {}'.format(index))
        plt.axis('off')
        plt.subplot(5, 4, 2 * number + 2)
        image = np.array(usps[noise_model][index]).reshape((16, 16))
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # For each class, take 300 noiseless samples for training and 50 noisy for testing
    # Show the image before and after denoising for the first 10 test samples
    for number in range(10):
        idx = np.random.choice(np.where(usps.target == number)[0], size=150, replace=False)
        train_idx, test_idx = idx[:100], idx[-50:]

        denoised = kPCA(usps.data[train_idx], usps[noise_model][test_idx]).obtain_preimages(20, 0.5)

        for i in range(10):
            plt.subplot(5, 4, 2 * i + 1)
            plt.imshow(usps[noise_model][test_idx[i]].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.title('Idx: {}'.format(test_idx[i]))
            plt.axis('off')
            plt.subplot(5, 4, 2 * i + 2)
            plt.imshow(denoised[i].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Consider class 3, take 300 noiseless samples for training and 50 noisy for testing
    # Show the denoising process when run with a variable number of features
    idx = np.random.choice(np.where(usps.target == 3)[0], size=150, replace=False)
    train_idx, test_idx = idx[:100], idx[-50:]

    for i, n_features in enumerate(range(2, 21, 2)):
        denoised = kPCA(usps.data[train_idx], usps[noise_model][test_idx]).obtain_preimages(n_features, 0.5)
        plt.subplot(5, 4, 2 * i + 1)
        plt.imshow(usps[noise_model][test_idx[0]].reshape((16, 16)), cmap='gray', interpolation='none')
        plt.title('Feat: {}'.format(n_features))
        plt.axis('off')
        plt.subplot(5, 4, 2 * i + 2)
        plt.imshow(denoised[0].reshape((16, 16)), cmap='gray', interpolation='none')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
