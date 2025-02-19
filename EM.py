import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def gaussian_mixture_density(x, means, variances, weights):
    density = np.zeros_like(x, dtype=float)
    for mean, var, weight in zip(means, variances, weights):
        density += weight * norm.pdf(x, loc=mean, scale=np.sqrt(var)).squeeze()
    return density


def score_fun_EM(ys, y_hat, k_hat, eps=1e-6):
    """
    ys: (N_ens, d);  y_hat: (d,)
    k_hat: scalar, number of clusters in k-NN
    """
    gmm = GaussianMixture(n_components=k_hat, random_state=42, reg_covar=eps)
    gmm.fit(ys)
    means = gmm.means_
    weights = gmm.weights_
    covariances = gmm.covariances_
    score = - gaussian_mixture_density(y_hat, means, covariances, weights) 
    return score

def summary_score_EM(data_ys, data_y_hat, k_hat):
    """
    data: (N_train, N_ens, d)
    k_hat: scalar
    """
    scores = []
    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):
        s = score_fun_EM(ys, y_hat, k_hat)
        scores.append(s)
    return np.array(scores)


# given k generated data, find prediction set with 1-alpha percent coverage rate on average
def inference_EM(ys, y_hat, k_hat, qt, eps=1e-6, grid_res=100):
    """
    ys: (N_ens, d)
    y_hat: (d,)
    k_hat: scalar
    qt: scalar
    """
    d = ys.shape[1]
    if len(ys) != k_hat:
        gmm = GaussianMixture(n_components=k_hat, random_state=42, reg_covar=eps)
        gmm.fit(ys)
        means = gmm.means_
        weights = gmm.weights_
        covariances = gmm.covariances_
    else:
        means = ys
        weights = [1/len(ys) for i in range(len(ys))]
        covariances = [eps * np.eye(d) for i in range(len(ys))]

    # compute score
    score = - gaussian_mixture_density(y_hat, means, covariances, weights)
    volume = get_coverage_length_brutal(ys, means, covariances, weights, qt, k_hat, grid_res)
    return score, volume

# compute volume for a single sample through MC
def get_coverage_length_brutal(ys, means, covariances, weights, qt, k_hat, grid_res=100):
    d = ys.shape[1]
    if d == 1:
        buffle = (np.max(ys[:,0]) - np.min(ys[:,0])) * 0.1
        x = np.linspace(np.min(ys[:,0])-buffle, np.max(ys[:,0])+buffle, grid_res)
        dens = - gaussian_mixture_density(x, means, covariances, weights)

        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= qt] = 1
        dens_new[dens > qt] = 0
        volume = (np.sum(dens_new)) * (x[1] - x[0])

    elif d == 2:
        buffle0 = (np.max(ys[:,0]) - np.min(ys[:,0])) * 0.1
        buffle1 = (np.max(ys[:,1]) - np.min(ys[:,1])) * 0.1
        x, y = np.meshgrid(np.linspace(np.min(ys[:,0])-buffle0, np.max(ys[:,0])+buffle0, grid_res), \
                           np.linspace(np.min(ys[:,1])-buffle1, np.max(ys[:,1])+buffle1, grid_res))
        pos = np.dstack((x, y))

        dens = - gaussian_mixture_density(pos, means, covariances, weights)
        
        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= qt] = 1
        dens_new[dens > qt] = 0
        volume = (np.sum(dens_new)) * (x[0,1] - x[0,0]) * (y[1,0] - y[0,0])
    return volume

def summary_inference_EM(data_ys, data_y_hat, k_hat, qt, grid_res=200):
    """
    data: (N_test, N_ens, d)
    k_hat: scalar, number of clusters in k-NN
    qt: scalar, quantile corresponding to the coverage rate
    """
    scores = []
    volumes = []

    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):
        score, volume = inference_EM(ys, y_hat, k_hat=k_hat, qt=qt, grid_res=grid_res)

        scores.append(score)
        volumes.append(volume)

    return np.array(scores), np.array(volumes)