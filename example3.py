import matplotlib.pyplot as plt
import numpy as np
import skimage
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA

from our_kpca import kPCA

# Fetch dataset and display info
usps = fetch_mldata('USPS')
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


def pca_denoising(training_data, test_data, n_components):
    "Performs linear PCA denoising using sklearn"
    pca = PCA(n_components)
    pca.fit(training_data)
    low_dim_representation = pca.transform(test_data)
    return pca.inverse_transform(low_dim_representation)


if __name__ == '__main__':
    for noise_model in ['gaussian', 's&p']:

        # Show the first occurrence of each number, with and without noise
        '''
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
        plt.suptitle('First occurrence of each number, with and without noise', fontsize=17)
        plt.tight_layout()
        plt.show()
        '''

        # For each class, take 300 noiseless samples for training and 10 noisy for testing
        # Show the image before and after denoising for the first 10 test samples
        '''
        for number in range(10):
            idx = np.random.choice(np.where(usps.target == number)[0], size=310, replace=False)
            train_idx, test_idx = idx[:300], idx[-10:]
            denoised = kPCA(usps.data[train_idx], usps[noise_model][test_idx]).obtain_preimages(20, 0.5)

            for i in range(10):
                plt.subplot(5, 4, 2 * i + 1)
                plt.imshow(usps[noise_model][test_idx[i]].reshape((16, 16)), cmap='gray', interpolation='none')
                plt.title('Idx: {}'.format(test_idx[i]))
                plt.axis('off')
                plt.subplot(5, 4, 2 * i + 2)
                plt.imshow(denoised[i].reshape((16, 16)), cmap='gray', interpolation='none')
                plt.axis('off')
            plt.suptitle('Trained only on {}, first 10 denoised samples'.format(number), fontsize=17)
            plt.tight_layout()
            plt.show()
        '''

        # For each class, take 300 noiseless samples for training and 1 noisy for testing.
        # Put all the samples together in a 10*300 training dataset and 10*1 testing dataset.
        # Show the image before and after denoising for the first occurrence of each class.
        '''
        train_idx = []
        test_idx = []
        for number in range(10):
            idx = np.random.choice(np.where(usps.target == number)[0], size=301, replace=False).tolist()
            train_idx += idx[:300]
            test_idx += idx[-1:]

        denoised = kPCA(usps.data[train_idx], usps[noise_model][test_idx]).obtain_preimages(20, 0.5)

        _, locs = np.unique(usps.target[test_idx], return_index=True)
        for plt_i, data_i in enumerate(locs):
            plt.subplot(5, 4, 2 * plt_i + 1)
            plt.imshow(usps[noise_model][test_idx[data_i]].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.title('Idx: {}'.format(test_idx[data_i]))
            plt.axis('off')
            plt.subplot(5, 4, 2 * plt_i + 2)
            plt.imshow(denoised[data_i].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.axis('off')
            plt.suptitle('Trained on all classes, first denoised sample of every class', fontsize=17)
        plt.tight_layout()
        plt.show()
        '''

        # Consider class 3, take 300 noiseless samples for training and 1 noisy for testing
        # Show the denoising process when run with a variable number of features
        '''
        idx = np.random.choice(np.where(usps.target == 3)[0], size=301, replace=False)
        train_idx, test_idx = idx[:300], idx[-1:]
        for i, n_features in enumerate(range(2, 21, 2)):
            denoised = kPCA(usps.data[train_idx], usps[noise_model][test_idx]).obtain_preimages(n_features, 0.5)
            plt.subplot(5, 4, 2 * i + 1)
            plt.imshow(usps[noise_model][test_idx[0]].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.title('Feat: {}'.format(n_features))
            plt.axis('off')
            plt.subplot(5, 4, 2 * i + 2)
            plt.imshow(denoised[0].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.axis('off')
        plt.suptitle('Trained only on 3, first denoised sample for different number of features'.format(number),
                     fontsize=17)
        plt.tight_layout()
        plt.show()
        '''

        # For each class, take 300 noiseless samples for training and 1 noisy for testing.
        # Put all the samples together in a 10*300 training dataset and 10*1 testing dataset.
        # Show the denoising process when run with a variable number of features.
        # Examples are made on class 3.
        '''
        train_idx = []
        test_idx = []
        for number in range(10):
            idx = np.random.choice(np.where(usps.target == number)[0], size=301, replace=False).tolist()
            train_idx += idx[:300]
            test_idx += idx[-1:]

        for i, n_features in enumerate(range(2, 21, 2)):
            denoised = kPCA(usps.data[train_idx], usps[noise_model][test_idx]).obtain_preimages(n_features, 0.5)
            plt.subplot(5, 4, 2 * i + 1)
            plt.imshow(usps[noise_model][test_idx[1 * 3]].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.title('Feat: {}'.format(n_features))
            plt.axis('off')
            plt.subplot(5, 4, 2 * i + 2)
            plt.imshow(denoised[1 * 3].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.axis('off')
        plt.suptitle('Trained on all classes, first denoised sample for different number of features'.format(number),
                     fontsize=17)
        plt.tight_layout()
        plt.show()
        '''

        # For each class, take 300 noiseless samples for training and 1 noisy for testing.
        # Put all the samples together in a 10*300 training dataset and 10*1 testing dataset.
        # Show the denoising process when run with a variable number of features.
        # Also show a comparison against a linear PCA.
        '''
        train_idx = []
        test_idx = []
        plt.figure(figsize=(10, 10))
        for number in range(10):
            idx = np.random.choice(np.where(usps.target == number)[0], size=301, replace=False).tolist()
            train_idx += idx[:300]
            test_idx += idx[-1:]
            plt.subplot(12, 10, number + 1)
            plt.imshow(usps.data[test_idx[number]].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.axis('off')
            plt.subplot(12, 10, 10 + number + 1)
            plt.imshow(usps[noise_model][test_idx[number]].reshape((16, 16)), cmap='gray', interpolation='none')
            plt.axis('off')

        for i, n_features in enumerate([1, 4, 16, 64, 256]):
            denoised = kPCA(usps.data[train_idx], usps[noise_model][test_idx]).obtain_preimages(n_features, 0.5)
            denoised_linear = pca_denoising(usps.data[train_idx], usps[noise_model][test_idx], n_features)
            for j, img in enumerate(denoised):
                plt.subplot(12, 10, 10 * (i + 5 + 2) + j + 1)
                plt.imshow(img.reshape((16, 16)), cmap='gray', interpolation='none')
                plt.axis('off')
            for j, img in enumerate(denoised_linear):
                plt.subplot(12, 10, 10 * (i + 2) + j + 1)
                plt.imshow(img.reshape((16, 16)), cmap='gray', interpolation='none')
                plt.axis('off')
        plt.subplots_adjust(wspace=0.03, hspace=0.03)
        plt.show()
        '''

    # For each class, take 300 noiseless samples.
    # Put all the samples together in a 10*300 training dataset.
    # Also take one noiseless sample of '3' for reconstruction.
    # Show the reconstruction process when run with a variable number of features, for both linear and kernel PCA
    train_idx = []
    for number in range(10):
        idx = np.random.choice(np.where(usps.target == number)[0], size=300, replace=False).tolist()
        train_idx += idx
    test_idx = np.where(usps.target == 3)[0][0]

    for n_features in range(1, 21):
        denoised = kPCA(usps.data[train_idx], usps.data[test_idx].reshape(1, -1), False).obtain_preimages(n_features, 0.5)
        denoised_linear = pca_denoising(usps.data[train_idx], usps.data[test_idx].reshape(1, -1), n_features)
        plt.subplot(6, 7, n_features + 7 * ((n_features - 1) // 7))
        plt.imshow(denoised.reshape((16, 16)), cmap='gray', interpolation='none')
        plt.title('{1:.2f} [K{0}] '.format(n_features, np.linalg.norm(denoised - usps.data[test_idx])))
        plt.axis('off')
        plt.subplot(6, 7, n_features + 7 * ((n_features - 1) // 7) + 7)
        plt.imshow(denoised_linear.reshape((16, 16)), cmap='gray', interpolation='none')
        plt.title('{1:.2f} [L{0}] '.format(n_features, np.linalg.norm(denoised_linear - usps.data[test_idx])))
        plt.axis('off')
    plt.subplot(6, 7, 42)
    plt.imshow(usps.data[test_idx].reshape((16, 16)), cmap='gray', interpolation='none')
    plt.axis('off')
    plt.show()
