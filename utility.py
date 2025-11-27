import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture


def get_overlap_length(intervals):
    temp_tuple = intervals
    temp_tuple.sort(key=lambda interval: interval[0])
    merged = [temp_tuple[0]]
    for current in temp_tuple:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    l = 0
    for x in merged:
        l += x[1]-x[0]

    return l 


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


def get_volume_1d(means, covariances, weights, quant_score):

    radius = [] 
    k = len(means)

    for i in range(k):
        sigma = covariances[i][0] ** 0.5
        radius.append(sigma * np.sqrt( max(0, 2*quant_score - np.log(2*np.pi) - 2*np.log(sigma/weights[i]))))

    volume = get_coverage_length_exact_1d(radius, means)

    return volume


# compute volume for a single sample through MC when d is larger than 2
def get_volume_nd(y_ens, means, covariances, weights, quant_score, grid_res=100, M=None):
    # y_ens: (N_ens, d)

    d = y_ens.shape[1]
    k = len(means)

    if d == 2:
        buffle0 = (np.max(y_ens[:, 0]) - np.min(y_ens[:, 0])) * 0.2
        buffle1 = (np.max(y_ens[:, 1]) - np.min(y_ens[:, 1])) * 0.2
        X_grid = np.linspace(np.min(y_ens[:, 0]) - buffle0, np.max(y_ens[:, 0]) + buffle0, grid_res)
        Y_grid = np.linspace(np.min(y_ens[:, 1]) - buffle1, np.max(y_ens[:, 1]) + buffle1, grid_res)
        x, y = np.meshgrid(X_grid, Y_grid)
        pos = np.dstack((x, y))
        dens = np.zeros((k, x.shape[0], x.shape[1]), float)

        for i in range(k):
            dens[i] = - (multivariate_normal.logpdf(pos, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        dens = np.min(dens, axis=0)
        
        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= quant_score] = 1
        dens_new[dens > quant_score] = 0
        volume = np.sum(dens_new) * (X_grid[1] - X_grid[0]) * (Y_grid[1] - Y_grid[0])

    else:
        if M is None:
            M = grid_res ** d
        buffle = [(np.max(y_ens[:, i]) - np.min(y_ens[:, i])) * 0.2 for i in range(d)]
        lower_bound = [np.min(y_ens[:, i]) - buffle[i] for i in range(d)]
        higher_bound = [np.max(y_ens[:, i]) + buffle[i] for i in range(d)]
        MC_data = np.random.uniform(low=lower_bound, high=higher_bound, size=(M, d))
        dens = np.zeros((k, M), float)

        for i in range(k):
            dens[i] = - (multivariate_normal.logpdf(MC_data, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        dens = np.min(dens, axis=0)

        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= quant_score] = 1
        dens_new[dens > quant_score] = 0
        volume = np.mean(dens_new) * np.prod(np.array(higher_bound) - np.array(lower_bound))

    return volume


def get_k_list(ens_size, d):
    # Reduce computing overhead searching less ks
    if (d >= 3) or (ens_size <= 10):
        k_list = [1, 2, 3, 4, 5, ens_size]
    else:
        k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if k_list[-1] == ens_size:
        return k_list
    else:
        return k_list + list(range(10 + 5, ens_size//2 + 5, 5)) + [ens_size]

