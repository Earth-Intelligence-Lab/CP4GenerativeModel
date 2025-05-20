import time
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import PCP
import KMean
from dataset import *
from flow_matching import *
from torch.utils.data import DataLoader
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture



dataset_name = 'Mengze'
data_path = './data/'
n_samples = 30
coverage = 0.9

Y_ens_calib, Y_calib, Y_ens_test, Y_test = get_togo_dataset(dataset_name, data_path)
Y_ens_calib = Y_ens_calib[:,:n_samples,:]
Y_ens_test = Y_ens_test[:,:n_samples,:]

# select 4 data to visualize
ens_visual = Y_ens_test[:4]
test_visual = Y_test[:4]


# KMeans
k_hat = 1
calib_scores = KMean.summary_score_KMeans(Y_ens_calib, Y_calib, k_hat=k_hat)
qt1 = np.quantile(calib_scores, coverage) 
test_scores, test_volumes = KMean.summary_inference_KMeans(Y_ens_calib, Y_calib, k_hat=k_hat, qt=qt1, grid_res=500)
print(f'k_hat: {k_hat}')
print(f'qt: {qt1}')
print(f'Test Coverage Rate: {np.mean(test_scores < qt1):.6f}')
print(f'Average Volume: {np.mean(test_volumes):.6f}')


# KMeans
k_hat = 2
calib_scores = KMean.summary_score_KMeans(Y_ens_calib, Y_calib, k_hat=k_hat)
qt2 = np.quantile(calib_scores, coverage) 
test_scores, test_volumes = KMean.summary_inference_KMeans(Y_ens_test, Y_test, k_hat=k_hat, qt=qt2)
print(f'k_hat: {k_hat}')
print(f'qt: {qt2}')
print(f'Test Coverage Rate: {np.mean(test_scores < qt2):.6f}')
print(f'Average Volume: {np.mean(test_volumes):.6f}')


# KMeans
k_hat = 3
calib_scores = KMean.summary_score_KMeans(Y_ens_calib, Y_calib, k_hat=k_hat)
qt3 = np.quantile(calib_scores, coverage) 
test_scores, test_volumes = KMean.summary_inference_KMeans(Y_ens_test, Y_test, k_hat=k_hat, qt=qt3)
print(f'k_hat: {k_hat}')
print(f'qt: {qt3}')
print(f'Test Coverage Rate: {np.mean(test_scores < qt3):.6f}')
print(f'Average Volume: {np.mean(test_volumes):.6f}')

# PCP
Y_hat = np.concatenate((Y_ens_calib, Y_ens_test), axis=0) # (n_batch, n_samples, dim_y)
Y_cal_test = np.concatenate((Y_calib, Y_test), axis=0).reshape(-1, 1, Y_ens_calib.shape[2]) # (n_batch, 1, dim_y)
# Ranking the samples by their average m-nearest neighbor distances, here we pick m=4.
# Compute pairwise distances between Y and Y_hat_ranked. Each row is a non-conformity score vector.
pcp_vcr = PCP.PCP_VCR(n_sample_K = n_samples,alpha=0.1,y_dim = Y_ens_calib.shape[2])
dist_matrix = pcp_vcr.compute_dist_matrix(Y_cal_test,Y_hat)
pcp_radius = pcp_vcr.pcp_radius(dist_matrix[:len(Y_calib)])
pcp_coverage = pcp_vcr.empirical_coverage(dist_matrix[len(Y_calib):],pcp_radius)
print(f'PCP Coverage Rate: {pcp_coverage:.6f}')

pcp_exact_volume1 = PCP.get_coverage_area_overlap(pcp_radius,Y_ens_test, M=1000)
print(f'PCP Average Volume1: {np.mean(pcp_exact_volume1):.6f}')
pcp_exact_volume1 = PCP.get_coverage_area_overlap(pcp_radius,Y_ens_test, M=5000)
print(f'PCP Average Volume1: {np.mean(pcp_exact_volume1):.6f}')
pcp_exact_volume1 = PCP.get_coverage_area_overlap(pcp_radius,Y_ens_test, M=10000)
print(f'PCP Average Volume1: {np.mean(pcp_exact_volume1):.6f}')


pcp_exact_volume2 = PCP.get_coverage_area_overlap_grid(pcp_radius,Y_ens_test, grid_res=1000)
print(f'PCP Average Volume2: {np.mean(pcp_exact_volume2):.6f}')
pcp_exact_volume2 = PCP.get_coverage_area_overlap_grid(pcp_radius,Y_ens_test, grid_res=5000)
print(f'PCP Average Volume2: {np.mean(pcp_exact_volume2):.6f}')
pcp_exact_volume2 = PCP.get_coverage_area_overlap_grid(pcp_radius,Y_ens_test, grid_res=10000)
print(f'PCP Average Volume2: {np.mean(pcp_exact_volume2):.6f}')


pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=1000)
print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}')
pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=5000)
print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}') 
pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=10000)
print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}') 
exit()

# pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=500)
# print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}')
# pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=1000)
# print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}')
# pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=5000)
# print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}')
# pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=10000)
# print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}')
# pcp_exact_volume3 = PCP.get_coverage_area_overlap_random(pcp_radius,Y_ens_test, M=15000)
# print(f'PCP Average Volume3: {np.mean(pcp_exact_volume3):.6f}')




# given k generated data, find prediction set with 90% percent coverage rate on average
def inference(ys, y_hat, k_hat, qt, radius, grid_res=500, eps=1e-6):

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
    buffle0 = (np.max(ys[:,0]) - np.min(ys[:,0])) * 0.4
    buffle1 = (np.max(ys[:,1]) - np.min(ys[:,1])) * 0.4
    buffle = [buffle0, buffle1]
    X_grid = np.linspace(np.min(ys[:,0])-buffle0, np.max(ys[:,0])+buffle0, grid_res)
    Y_grid = np.linspace(np.min(ys[:,1])-buffle1, np.max(ys[:,1])+buffle1, grid_res)
    x, y = np.meshgrid(X_grid, Y_grid)
    pos = np.dstack((x, y))
    dens = np.zeros((k_hat, x.shape[0], x.shape[1]), float)

    for i in range(k_hat):
        dens[i] = - (multivariate_normal.logpdf(pos, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
    dens = np.min(dens, axis=0)

    dens_new = np.zeros(dens.shape) * np.nan
    dens_new[dens <= qt] = 1
    dens_new[dens > qt] = 0
    km_volume = (np.sum(dens_new)) * (X_grid[1] - X_grid[0]) * (Y_grid[1] - Y_grid[0])
    # print('km_volume', km_volume)
    
    grid_data = np.dstack((x, y)).reshape(grid_res**d, d, 1)
    dens_pcp = np.any(np.linalg.norm(grid_data-ys.T,axis=1)<=radius,axis=1).reshape(x.shape[0], x.shape[1])
    coverage_ratio = np.mean(dens_pcp)
    # print('coverage_ratio', coverage_ratio)
    volume = np.prod(np.array([np.max(ys[:,i])+buffle[i] for i in range(d)]) - np.array([np.min(ys[:,i])-buffle[i] for i in range(d)]))
    # print('volume', volume)
    pcp_volume = coverage_ratio * volume
    # print('pcp_volume', pcp_volume)

    # compute score
    scores = []
    for i in range(k_hat):
        score = - (multivariate_normal.logpdf(y_hat, mean=means[i], cov=covariances[i], allow_singular=True) + np.log(weights[i]))
        scores.append(score)
    volume_ = KMean.get_coverage_length_brutal(ys, means, covariances, weights, qt, k_hat, grid_res)
    # print('volume_', volume_)
    return min(scores), dens_new, dens_pcp, x, y, km_volume, pcp_volume


def visualize_data(ens_visual, test_visual, k_hat, qt, pcp_radius):


    # Create 4x2 subplot
    fig, axes = plt.subplots(4, figsize=(6, 20))

    # Plot each test case
    for i in range(4):
        ys = ens_visual[i]
        y_hat = test_visual[i]
        km_score, km_dens, km_dens_pcp, km_x, km_y, km_volume, pcp_volume = inference(ys, y_hat, k_hat, qt, pcp_radius)

        # Plot the data points
        buffle0 = (np.max(ys[:,0]) - np.min(ys[:,0])) * 0.2
        buffle1 = (np.max(ys[:,1]) - np.min(ys[:,1])) * 0.2
        axes[i].scatter(ys[:,0], ys[:,1], marker='o', color = 'red', s=10, zorder = 2.5)
        axes[i].scatter(y_hat[0], y_hat[1], marker='^', color = 'orange', s=100, zorder = 2.5)
        axes[i].contour(km_x, km_y, km_dens, [0.5], colors='r')
        axes[i].set_xlim(np.min(km_x)-buffle0/2, np.max(km_x)+buffle0/2)
        axes[i].set_ylim(np.min(km_y)-buffle1/2, np.max(km_y)+buffle1/2)

        # Plot circles of radius pcp_radius around each ensemble point
        for j in range(ys.shape[0]):
            circle = plt.Circle((ys[j,0], ys[j,1]), pcp_radius, fill=False, color='blue', alpha=0.3)
            axes[i].add_patch(circle)
        pcp_volume_ = PCP.get_coverage_area_overlap_grid(pcp_radius, ys.reshape(1, -1, 2))[0]

        axes[i].set_title(f'km Volume: {km_volume:.6f}, pcp Volume: {pcp_volume:.6f}')
    # Add super title showing k_hat and method labels
    fig.suptitle(f'k_hat = {k_hat}', fontsize=16, y=0.95)
    plt.show()
    return


k_hat = 1
visualize_data(ens_visual, test_visual, k_hat, qt1, pcp_radius)


k_hat = 2
visualize_data(ens_visual, test_visual, k_hat, qt2, pcp_radius)


k_hat = 3
visualize_data(ens_visual, test_visual, k_hat, qt3, pcp_radius)