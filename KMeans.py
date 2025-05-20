import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture


def score_fun_KMeans(ys, y_hat, k_hat, eps=1e-6):

    # ys: (N_ens, d);  y_hat: (d,)
    # k_hat: scalar, number of clusters in k-NN

    d = ys.shape[1]

    if len(ys) != k_hat: # number of clusters less than number of ensemble members
      kmeans = KMeans(n_clusters=k_hat, n_init='auto', random_state=0).fit(ys)
      means = kmeans.cluster_centers_
      weights = [np.mean(kmeans.labels_ == i) for i in range(k_hat)]
      covariances = [np.cov(ys[kmeans.labels_ == i].T) + eps * np.eye(d) if weights[i] * len(ys) > 1 else eps * np.eye(d) for i in range(k_hat)]
    else: # every ensemble member is a cluster
      means = ys
      weights = [1/len(ys)] * len(ys)
      covariances = [eps * np.eye(d) for i in range(len(ys))]

    distances = []
    for i in range(k_hat):
        distance = - (multivariate_normal.logpdf(y_hat, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        distances.append(distance)

    return min(distances)


def summary_score_KMeans(data_ys, data_y_hat, k_hat):

    # data: (N_train, N_ens, d)
    # k_hat: scalar

    scores = []
    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):
        s = score_fun_KMeans(ys, y_hat, k_hat)
        scores.append(s)

    return scores


def summary_score_KMeans_adaptive(data_ys, data_y_hat, weight_th, max_k=10):

    # data: (N_train, N_ens, d)
    # weight_th: scalar, threshold for the weight of the cluster

    scores = []
    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):

        bgmm = BayesianGaussianMixture(n_components=max_k, random_state=42)
        bgmm.fit(ys)

        weights = bgmm.weights_
        k_hat = max(1, np.sum(weights > weight_th))
        
        s = score_fun_KMeans(ys, y_hat, k_hat)
        scores.append(s)

    return scores


# given k generated data, find prediction set with 1-alpha percent coverage rate on average
def inference_KMeans(ys, y_hat, k_hat, qt, eps=1e-6, grid_res=100):

    # ys: (N_ens, d)
    # y_hat: (d,)
    # k_hat: scalar
    # qt: scalar

    d = ys.shape[1]

    if len(ys) != k_hat:
      kmeans = KMeans(n_clusters=k_hat, n_init='auto', random_state=0).fit(ys)
      means = kmeans.cluster_centers_
      weights = [np.mean(kmeans.labels_ == i) for i in range(k_hat)]
      covariances = [np.cov(ys[kmeans.labels_ == i].T) + eps * np.eye(d) if weights[i] * len(ys) > 1 else eps * np.eye(d) for i in range(k_hat)]
    else:
      means = ys
      weights = [1/len(ys) for i in range(len(ys))]
      covariances = [eps * np.eye(d) for i in range(len(ys))]

    # Gaussian KDE prediction sets
    if d == 2:
        buffle = (np.max(ys[:,0]) - np.min(ys[:,0])) * 0.1
        x, y = np.meshgrid(np.linspace(np.min(ys[:,0])-buffle, np.max(ys[:,0])+buffle, grid_res), np.linspace(np.min(ys[:,1])-buffle, np.max(ys[:,1])+buffle, grid_res))
        pos = np.dstack((x, y))
        dens = np.zeros((k_hat, x.shape[0], x.shape[1]), float)

        for i in range(k_hat):
            dens[i] = - (multivariate_normal.logpdf(pos, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        dens = np.min(dens, axis=0)
        
        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= qt] = 1
        dens_new[dens > qt] = 0

        # compute score
        scores = []
        for i in range(k_hat):
            score = - (multivariate_normal.logpdf(y_hat, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
            scores.append(score)

        volume = (np.sum(dens_new)) * (x[0,1] - x[0,0]) * (y[1,0] - y[0,0])

    elif d == 1:
        buffle = (np.max(ys[:,0]) - np.min(ys[:,0])) * 0.1
        x = np.linspace(np.min(ys[:,0])-buffle, np.max(ys[:,0])+buffle, grid_res)
        dens = np.zeros((k_hat, x.shape[0]), float)
        for i in range(k_hat):
            dens[i] = - (multivariate_normal.logpdf(x, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        dens = np.min(dens, axis=0) # min over k_hat----we could also take the summation?

        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= qt] = 1
        dens_new[dens > qt] = 0

        # compute score
        scores = []
        for i in range(k_hat):
            score = - (multivariate_normal.logpdf(y_hat, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
            scores.append(score)

        volume = (np.sum(dens_new)) * (x[1] - x[0])

    return min(scores), dens_new, volume


def summary_inference_KMeans(data_ys, data_y_hat, k_hat, qt, grid_res=200):

    # data: (N_test, N_ens, d)
    # k_hat: scalar, number of clusters in k-NN
    # qt: scalar, quantile corresponding to the coverage rate

    scores = []
    volumes = []

    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):
        score, dens, volume = inference_KMeans(ys, y_hat, k_hat=k_hat, qt=qt, grid_res=grid_res)

        scores.append(score)
        volumes.append(volume)

    return np.array(scores), np.array(volumes)


def summary_inference_KMeans_adaptive(data_ys, data_y_hat, weight_th, qt, max_k=10, grid_res=200):

    # data_ys: (N_test, N_ens, d)
    # data_y_hat: (N_test, d)
    # weight_th: scalar, threshold for the weight of the cluster
    # qt: scalar, quantile corresponding to the coverage rate
    # k_max: scalar, maximum number of clusters

    scores = []
    volumes = []

    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):

        bgmm = BayesianGaussianMixture(n_components=max_k, random_state=42)
        bgmm.fit(ys)

        weights = bgmm.weights_
        k_hat = max(1, np.sum(weights > weight_th))        

        score, dens, volume = inference_KMeans(ys, y_hat, k_hat=k_hat, qt=qt, grid_res=grid_res)

        scores.append(score)
        volumes.append(volume)

    return np.array(scores), np.array(volumes)



