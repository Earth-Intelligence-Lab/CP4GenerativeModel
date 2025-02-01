import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

def score_fun(ens, k_hat, eps=1e-6):

    # ens: (N_ens, d)
    # k_hat: scalar
    d = ens.shape[1]
    ys, y_hat = ens[:-1, :], ens[[-1], :]

    if len(ys) != k_hat:
      kmeans = KMeans(n_clusters=k_hat, random_state=0).fit(ys)
      means = kmeans.cluster_centers_
      weights = [np.mean(kmeans.labels_ == i) for i in range(k_hat)]
      covariances = [np.cov(ys[kmeans.labels_ == i].T) + eps * np.eye(d) if weights[i] * len(ys) > 1 else eps * np.eye(d) for i in range(k_hat)]
    else:
      means = ys
      weights = [1/len(ys) for i in range(len(ys))]
      covariances = [eps * np.eye(d) for i in range(len(ys))]

    #print(covariances)
    #print(weights)

    distances = []
    for i in range(k_hat):
        distance = - (multivariate_normal.logpdf(y_hat, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        distances.append(distance)

    return min(distances)


def summary_score(data, k_hat):

    # data: (N_train, N_ens, d)
    # k_hat: scalar

    scores = []
    for ens in tqdm(data):
        s = score_fun(ens, k_hat)
        scores.append(s)
    return scores


# given k generated data, find prediction set with 1-alpha% percent coverage rate on average
def inference(ys, y_hat, k_hat, qt, eps=1e-6, grid_res=100, buffle=3):

    # ys: (N_ens, d)
    # y_hat: (1, d)
    # k_hat: scalar
    # qt: scalar

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

    # Gaussian KDE prediction sets
    x, y = np.meshgrid(np.linspace(np.min(ys[:,0])-buffle, np.max(ys[:,0])+buffle, grid_res), np.linspace(np.min(ys[:,1])-buffle, np.max(ys[:,1])+buffle, grid_res))
    pos = np.dstack((x, y))
    dens = np.zeros((k_hat, x.shape[0], x.shape[1]), float)

    for i in range(k_hat):
        dens[i] = - (multivariate_normal.logpdf(pos, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
    dens = np.min(dens, axis=0)

    dens[dens < qt] = 0
    dens[dens >= qt] = 1

    # compute score
    scores = []
    for i in range(k_hat):
        score = - (multivariate_normal.logpdf(y_hat, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        scores.append(score)

    return min(scores), dens, x, y



def summary_inference(data, k_hat, qt, grid_res=200, buffle=3):

    # data: (N_test, N_ens, d)
    # k_hat: scalar
    # qt: scalar

    scores = []
    volumes = []

    for ens in tqdm(data):
        score, dens, x, y = inference(ys=ens[:-1, :], y_hat=ens[[-1], :], k_hat=k_hat, qt=qt, grid_res=grid_res, buffle=buffle)
        volume = (x.shape[0] * x.shape[1] - np.sum(dens)) * (x[0,1] - x[0,0]) * (y[1,0] - y[0,0])

        scores.append(score)
        volumes.append(volume)

    return np.array(scores), np.array(volumes)