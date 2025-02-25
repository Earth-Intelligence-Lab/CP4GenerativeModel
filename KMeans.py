import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import gamma
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal


# compute score for single sample
def score_fun_KMeans(ys, y_hat, k_hat, eps=1e-6):
    """
    ys: (N_ens, d);  y_hat: (d,)
    k_hat: scalar, number of clusters in k-NN
    """
    d = ys.shape[1]
    if len(ys) != k_hat: # number of clusters less than number of ensemble members
      kmeans = KMeans(n_clusters=k_hat, random_state=0).fit(ys)
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


# compute score for multiple samples
def summary_score_KMeans(data_ys, data_y_hat, k_hat):
    # data: (N_train, N_ens, d)
    # k_hat: scalar
    scores = []
    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):
        s = score_fun_KMeans(ys, y_hat, k_hat)
        scores.append(s)
    return scores


# find score & prediction set volume for a single sample with 1-alpha percent coverage rate on average
def inference_KMeans(ys, y_hat, k_hat, qt, eps=1e-6, grid_res=100):
    """
    ys: (N_ens, d)
    y_hat: (d,)
    k_hat: scalar
    qt: scalar
    """
    d = ys.shape[1]
    if len(ys) != k_hat:
      kmeans = KMeans(n_clusters=k_hat, random_state=0).fit(ys)
      means = kmeans.cluster_centers_
      weights = [np.mean(kmeans.labels_ == i) for i in range(k_hat)]
      covariances = [np.cov(ys[kmeans.labels_ == i].T) + eps * np.eye(d) if weights[i] * len(ys) > 1 else eps * np.eye(d) for i in range(k_hat)]
    else:
      means = ys
      weights = [1/len(ys) for i in range(len(ys))]
      covariances = [eps * np.eye(d) for i in range(len(ys))]

    # compute score for the sample
    scores = []
    for i in range(k_hat):
        score = - (multivariate_normal.logpdf(y_hat, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        scores.append(score)
    score = min(scores)
    # print('weights: ', weights)
    # print('covariances: ', covariances)
    # compute volume for the prediction set
    if d == 1:
        radius = []
        for i in range(k_hat):
            sigma = covariances[i][0]**0.5
            # print(2*qt - np.log( 2*np.pi) - 2*np.log(sigma / weights[i] ))
            radius.append(sigma * np.sqrt( max( 0, 2*qt - np.log( 2*np.pi) - 2*np.log(sigma / weights[i]) ) ) )
        volume = get_coverage_length_exact(radius, means)
    elif d == 2:
        volume = get_coverage_length_brutal(ys, means, covariances, weights, qt, k_hat, grid_res)

    return score, volume


# compute volume for a single sample through MC
def get_coverage_length_brutal(ys, means, covariances, weights, qt, k_hat, grid_res=100):
    d = ys.shape[1]
    if d == 1:
        buffle = (np.max(ys[:,0]) - np.min(ys[:,0])) * 0.1
        x = np.linspace(np.min(ys[:,0])-buffle, np.max(ys[:,0])+buffle, grid_res)
        dens = np.zeros((k_hat, x.shape[0]), float)
        for i in range(k_hat):
            dens[i] = - (multivariate_normal.logpdf(x, mean=means[i], cov=covariances[i], allow_singular=True)\
                      + np.log(weights[i]))
        dens = np.min(dens, axis=0) # min over k_hat----we could also take the summation?

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
        dens = np.zeros((k_hat, x.shape[0], x.shape[1]), float)

        for i in range(k_hat):
            dens[i] = - (multivariate_normal.logpdf(pos, mean=means[i], cov=covariances[i], allow_singular=True)\
                      + np.log(weights[i]))
        dens = np.min(dens, axis=0)
        
        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= qt] = 1
        dens_new[dens > qt] = 0
        volume = (np.sum(dens_new)) * (x[0,1] - x[0,0]) * (y[1,0] - y[0,0])
    else:
        M=100**d
        buffle = [(np.max(ys[:,i]) - np.min(ys[:,i])) * 0.1 for i in range(d)]
        lower_bound = [np.min(ys[:,i]) - buffle[i] for i in range(d)]
        higher_bound = [np.max(ys[:,i]) + buffle[i] for i in range(d)]
        MC_data = np.random.uniform(low=lower_bound, high=higher_bound, size=(M,d))
        dens = np.zeros((k_hat, MC_data.shape[0]), float)
        for i in range(k_hat):
            dens[i] = - (multivariate_normal.logpdf(MC_data, mean=means[i], cov=covariances[i], allow_singular=True)\
                      + np.log(weights[i]))
        dens = np.min(dens, axis=0)

        dens_new = np.zeros(dens.shape) * np.nan
        dens_new[dens <= qt] = 1
        dens_new[dens > qt] = 0
        volume = (np.sum(dens_new)) * np.prod(np.array(higher_bound) - np.array(lower_bound))
    return volume


# compute coverage length for a single sample exactly
def get_coverage_length_exact(radius,Y_test):
    """
    Y_test in the shape of (n_sample, d)
    """
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
        l=0
        for x in merged:
            l+= x[1]-x[0]
        return l 
    
    n_sample = Y_test.shape[0]
    if isinstance(radius, float):
        radius = [radius]*n_sample

    I = []
    for j in range(n_sample):
        l = (Y_test[j]-radius[j])
        u = (Y_test[j]+radius[j])
        I.append([l,u])
    coverage_length=get_overlap_length(I)
    return coverage_length


def get_coverage_volume_overlap(means, covariances, weights, k_hat, qt):
    volumes = []
    d = means.shape[1]
    for i in range(k_hat):
        det = np.linalg.det(covariances[i])
        r = np.sqrt( 2*qt - d*np.log(2*np.pi)-np.log(det) + 2*np.log(weights[i]) )
        volumes.append(np.pi**(d/2) * r**d / gamma(d/2+1) * np.sqrt(det))
    return np.sum(volumes)


# compute score & prediction set volume for multiple samples
def summary_inference_KMeans(data_ys, data_y_hat, k_hat, qt, grid_res=200):
    """
    data: (N_test, N_ens, d)
    k_hat: scalar, number of clusters in k-NN
    qt: scalar, quantile corresponding to the coverage rate
    """
    scores = []
    volumes = []

    for idx, (ys, y_hat) in enumerate(zip(data_ys, data_y_hat)):
        score, volume = inference_KMeans(ys, y_hat, k_hat=k_hat, qt=qt, grid_res=grid_res)

        scores.append(score)
        volumes.append(volume)

    return np.array(scores), np.array(volumes)