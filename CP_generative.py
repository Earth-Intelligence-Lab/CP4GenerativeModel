import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from utility import *


def fit_KMeans(y_ens, k, eps=1e-6):
    # y_ens: (N_ens, d)
    # k: scalar, number of clusters in k-NN

    d = y_ens.shape[1]

    if len(y_ens) > k: 
    # number of clusters less than number of ensemble members
      kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(y_ens)
      means = kmeans.cluster_centers_
      weights = [np.mean(kmeans.labels_ == i) for i in range(k)]
      covariances = [np.cov(Y_ens[kmeans.labels_ == i].T) + eps * np.eye(d) if weights[i] * len(Y_ens) > 1 else eps * np.eye(d) for i in range(k)]
    
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


def get_coverage_length_exact_1d(radius, centers):
    # centers: (k, 1)
    # radius: scalar or (k, 1)

    n_sample = len(centers)
    if isinstance(radius, float):
        radius = [radius] * n_sample

    I = []
    for j in range(n_sample):
        l = (centers[j] - radius[j])
        u = (centers[j] + radius[j])
        I.append([l, u])
    
    coverage_length = get_overlap_length(I)

    return coverage_length


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
        pass
            