# -*- coding: utf-8 -*-
"""
Sample results :
with our own implementation
gamma	1		2		3		4		5		6		7		8		9
0.05	8592.64	8669.87	3260.86	1937.01	930.31	717.78	237.82	73.03	77.33
0.1		1676.85	992.82	1291.58	734.77	500.27	174.66	112.50	33.48	90.59
0.2		4.71	1.46	5.60	3.97	12.32	5.33	35.57	49.11	67.07
0.4		nan	    0.87	1.18	2.03	2.50	3.78	4.38	6.38	6.26
0.8		nan	    0.50	0.56	0.70	0.89	0.95	1.07	1.34	1.79


with Library's implementation
gamma	1		2		3		4		5		6		7		8		9
0.05	0.84	0.83	0.74	0.58	0.32	0.27	0.15	0.43	3.21
0.1		0.85	0.53	0.72	0.66	0.52	0.24	0.19	0.70	3.12
0.2		0.94	0.86	1.04	0.99	1.41	3.92	1.73	6.47	2.57
0.4		1.03	1.13	1.27	1.62	1.88	2.10	2.28	2.77	2.46
0.8		1.04	1.14	1.20	1.24	1.38	1.41	1.45	1.53	1.70

Possible explanations for the results :
NAN results appear when the denominator in eq.(10) is 0. Maybe restart from another starting point ?
I used the original test data as starting point in the iteration as suggested in the article.
Our result seem to follow roughly the same variation as in the paper, one result for differences could be the use of the
library PCA ?
The library result seem to perform a lot less well than our implementation / the one in the paper : probably their
implementation is for re-construction rather than de-noising, and not starting the iteration from the test-data can lead
to other fixed points of (10) than the one closest to our data points, giving bad results especially when gamma is low.
"""

import matplotlib.pyplot as plt
import numpy as np
import gaussianPCA
from scipy.spatial.distance import cdist
from scipy import exp
from sklearn.decomposition import PCA, KernelPCA

MSEOurImpl = {}
MSELib = {}

for sigma in [0.05,0.1,0.2,0.4,0.8]:
    gam = 1/(10*2*sigma*sigma)
    MSEOurImpl[sigma] = {}
    MSELib[sigma] = {}
    for n in range(1,10) :

        # generate 11 gaussians :
        # pick their centers :
        centers = np.random.uniform(low=-1.0, high=1.0, size=(11,10))
        # construct the train data :
        train = np.random.multivariate_normal(mean=centers[0], cov=sigma*sigma*np.eye(10), size=100)
        for i in range(1,11):
            train = np.concatenate((train, np.random.multivariate_normal(mean=centers[i], cov=sigma*sigma*np.eye(10), size=100)), axis=0)
        # construct the test data
        test = np.random.multivariate_normal(mean=centers[0], cov=sigma*sigma*np.eye(10), size=33)
        for i in range(1,11):
            test = np.concatenate((test, np.random.multivariate_normal(mean=centers[i], cov=sigma*sigma*np.eye(10), size=33)), axis=0)

        # using libraries :
        kpca = KernelPCA(n_components=n, kernel="rbf", fit_inverse_transform=True, gamma=gam)
        kpca.fit(train)
        test_transK = kpca.transform(test)
        ZK = kpca.inverse_transform(test_transK)
        pca = PCA(n_components=n)
        pca.fit(train)
        test_transL = pca.transform(test)
        ZL = pca.inverse_transform(test_transL)



        ### Using our code (gaussianPCA is inspired from the simple impl. we found) :
        # start by finding alphas using slightly modified version of the online code (this is SLOW)
        alpha = gaussianPCA.stepwise_kpca(train, gam, n)
        ZOurImpl = []
        # For each of the test points :
        for x in test :
            # Deduce the betaKs
            k = cdist([x],train, 'sqeuclidean')
            k = exp(-gam * k)
            beta = np.sum(alpha*k.T, axis=0)
            # deduce the gamma_i :
            gamma = np.sum(alpha*beta, axis=1)
            # pick x as the starting point for eq. (10) iterations
            newZ = x
            z = np.zeros((10))
            # iterate as in equation(10) until sufficient convergence
            while np.max(z-newZ)>0.00001 :
                z = newZ
                zcoeff = cdist([z],train, 'sqeuclidean')
                zcoeff = exp(-gam * zcoeff)
                zcoeff = zcoeff * gamma.T
                newZ = np.sum(train*zcoeff.T, axis=0)
                newZ = newZ / np.sum(zcoeff)
            ZOurImpl.append(newZ)

        ZOurImpl = np.array(ZOurImpl)


        ## Compute the MSE ##
        # Substract the centers :
        for i in range(363):
            ZK[i] = ZK[i] - centers[i//33]
            ZL[i] = ZL[i] - centers[i//33]
            ZOurImpl[i] = ZOurImpl[i] - centers[i//33]

        # add it to the dictionnaty
        MSEOurImpl[sigma][n] = (ZL**2).mean(axis=None)/(ZOurImpl**2).mean(axis=None)
        MSELib[sigma][n] = (ZL**2).mean(axis=None)/(ZK**2).mean(axis=None)


### Printing ###
print("ratios for our impl :")
print(MSEOurImpl)
print("ratios for the lib :")
print(MSELib)

print('gamma\t1\t\t2\t\t3\t\t4\t\t5\t\t6\t\t7\t\t8\t\t9\t\t')
for gamma in MSEOurImpl :
    print('{}\t'.format(gamma), end="")
    if (gamma != 0.05):
        print('\t', end="")
    for n in MSEOurImpl[gamma] :
        print('{0:.2f}\t'.format(MSEOurImpl[gamma][n]), end="")
    print('\n', end="")
print('\n')

print('gamma\t1\t\t2\t\t3\t\t4\t\t5\t\t6\t\t7\t\t8\t\t9\t\t')
for gamma in MSELib :
    print('{}\t'.format(gamma), end="")
    if (gamma != 0.05):
        print('\t', end="")
    for n in MSELib[gamma] :
        print('{0:.2f}\t'.format(MSELib[gamma][n]), end="")
    print('\n', end="")