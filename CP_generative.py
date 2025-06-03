import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

from utility import *


def find_k(y_ens, max_k, method='BIC', **kwargs):
    # y_ens: (N_ens, d)
    # max_k: scalar, maximum number of clusters
    # method: string, method to find the number of clusters
    # **kwargs: additional arguments

    if method == 'BIC':
        # Bayesian Information Criterion
        bics = []
        for k in range(1, max_k + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
            gmm.fit(y_ens)
            bics.append(gmm.bic(y_ens))

        return np.argmin(bics) + 1

    if method == 'BGMM':
        # Bayesian Gaussian Mixture Model
        bgmm = BayesianGaussianMixture(n_components=max_k, random_state=42)
        bgmm.fit(y_ens)
        ws = bgmm.weights_
        k = max(1, np.sum(ws >= kwargs['w_thred']))

        return k


def fit_KMeans(y_ens, k, eps=1e-6):
    # y_ens: (N_ens, d)
    # k: scalar, number of clusters in k-NN

    d = y_ens.shape[1]

    if len(y_ens) > k: 
    # number of clusters less than number of ensemble members
      kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(y_ens)
      means = kmeans.cluster_centers_
      weights = [np.mean(kmeans.labels_ == i) for i in range(k)]
      covariances = [np.cov(y_ens[kmeans.labels_ == i].T) + eps * np.eye(d) if weights[i] * len(y_ens) > 1 else eps * np.eye(d) for i in range(k)]
    
    if len(y_ens) == k:
    # every ensemble member is a cluster
      means = y_ens
      weights = [1/len(y_ens)] * len(y_ens)
      covariances = [eps * np.eye(d) for i in range(len(y_ens))]

    return means, covariances, weights


def score_fun_KMeans(y, means, covariances, weights):
    # y: (d,)

    k = len(means)

    distances = []
    for i in range(k):
        distance = - (multivariate_normal.logpdf(y, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        distances.append(distance)

    return min(distances)


class CPGen:
    def __init__(self, args, k):
        self.args = args
        self.k = k
        self.coverage = args.coverage

    def fit(self, Y_ens, Y):
        # Y_ens: (N_batch, N_ens, d)
        # Y: (N_batch, d)

        scores = []
        for idx, (y_ens, y) in enumerate(zip(Y_ens, Y)):
            means, covariances, weights = fit_KMeans(y_ens, self.k)
            s = score_fun_KMeans(y, means, covariances, weights)
            scores.append(s)

        scores = np.array(scores)
        self.quant_score = np.quantile(scores, self.coverage)

    def predict(self, Y_ens, Y):
        # Y_ens: (N_batch, N_ens, d)
        # Y: (N_batch, d)
        
        d = Y.shape[1]

        scores = []
        volumes = []
        ks = []

        for idx, (y_ens, y) in enumerate(zip(Y_ens, Y)):
            means, covariances, weights = fit_KMeans(y_ens, self.k)
            s = score_fun_KMeans(y, means, covariances, weights)
            scores.append(s)

            if d == 1:
                v = get_volume_1d(means, covariances, weights, self.quant_score)
            else:
                v = get_volume_nd(y_ens, means, covariances, weights, self.quant_score)
            
            volumes.append(v)
            ks.append(len(means))

        return np.array(scores), np.array(volumes), np.array(ks)


class CPGen_Adaptive:
    def __init__(self, args, **kwargs):
        self.args = args
        self.coverage = args.coverage
        self.max_k = args.max_k
        self.k_method = args.k_method
        self.kwargs = kwargs

    def fit(self, Y_ens, Y):
        # Y_ens: (N_batch, N_ens, d)
        # Y: (N_batch, d)

        scores = []
        for idx, (y_ens, y) in enumerate(zip(Y_ens, Y)):

            k = find_k(y_ens, self.max_k, method=self.k_method, **self.kwargs)
            means, covariances, weights = fit_KMeans(y_ens, k)
            s = score_fun_KMeans(y, means, covariances, weights)
            scores.append(s)

        scores = np.array(scores)
        self.quant_score = np.quantile(scores, self.coverage)

    def predict(self, Y_ens, Y):
        # Y_ens: (N_batch, N_ens, d)
        # Y: (N_batch, d)
        
        d = Y.shape[1]

        scores = []
        volumes = []
        ks = []

        for idx, (y_ens, y) in enumerate(zip(Y_ens, Y)):

            k = find_k(y_ens, self.max_k, method=self.k_method, **self.kwargs)
            means, covariances, weights = fit_KMeans(y_ens, k)
            s = score_fun_KMeans(y, means, covariances, weights)
            scores.append(s)

            if d == 1:
                v = get_volume_1d(means, covariances, weights, self.quant_score)
            else:
                v = get_volume_nd(y_ens, means, covariances, weights, self.quant_score)
            
            volumes.append(v)
            ks.append(k)

        return np.array(scores), np.array(volumes), np.array(ks)






            
