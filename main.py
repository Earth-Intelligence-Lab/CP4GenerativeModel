import os
import sys
from tqdm import tqdm
import torch
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

import PCP
import KMeans
from dataset import *
from flow_matching import *
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def run(args):
  
    # Set random seed
    random_state = args.seed
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    # Get togo data list
    togo_list = ['Mengze', 'synthetic_normal']

    # Get data set
    dataset_name = args.dataset

    if dataset_name not in togo_list:

        X, Y = get_dataset(dataset_name, data_path=args.data_path)
        N = X.shape[0]

        train, calib, test = np.split(range(N), [int(.6 * N), int(.8 * N), ])
        print(f'train size: {len(train)}, calib size: {len(calib)}, test size: {len(test)}', flush=True)

        # Extract train, calib, and test subsets
        X_train, Y_train = X[train], Y[train]
        X_calib, Y_calib = X[calib], Y[calib]
        X_test, Y_test = X[test], Y[test]

        # Standardize the data
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_calib = x_scaler.transform(X_calib)
        X_test = x_scaler.transform(X_test)

        y_scaler = StandardScaler()
        Y_train = y_scaler.fit_transform(Y_train)
        Y_calib = y_scaler.transform(Y_calib)
        Y_test = y_scaler.transform(Y_test)

        # Create Dataset objects
        train_dataset = DatasetTensor(X_train, Y_train)
        calib_dataset = DatasetTensor(X_calib, Y_calib)
        test_dataset = DatasetTensor(X_test, Y_test)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        calib_loader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Construct Generative Model
        linear_alpha = LinearAlpha()
        squareroot_beta = SquareRootBeta()
        gaussian_path = GaussianPath(linear_alpha, squareroot_beta)

        model = FlowMatchingNet(input_dim=Y.shape[1], condition_dim=X.shape[1], hidden_dim=args.hidden_dim).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train Generative Model
        train_flow_matching(model, gaussian_path, train_loader, optimizer, args.n_epochs, args.device)

        # Sample Generative Model
        calib_samples, calib_conditions = generate_samples_for_dataset(model, gaussian_path, calib_loader, args.n_samples, args.timesteps, args.device)
        test_samples, test_conditions = generate_samples_for_dataset(model, gaussian_path, test_loader, args.n_samples, args.timesteps, args.device)
        # (n_batch, n_samples, dim_y); (n_batch, n_sample, dim_x)
        
        # Denormalize the samples and conditions
        calib_samples = y_scaler.inverse_transform(calib_samples.reshape(-1, calib_samples.shape[-1])).reshape(calib_samples.shape)
        calib_conditions = x_scaler.inverse_transform(calib_conditions.reshape(-1, calib_conditions.shape[-1])).reshape(calib_conditions.shape)
        test_samples = y_scaler.inverse_transform(test_samples.reshape(-1, test_samples.shape[-1])).reshape(test_samples.shape)
        test_conditions = x_scaler.inverse_transform(test_conditions.reshape(-1, test_conditions.shape[-1])).reshape(test_conditions.shape)
        # (n_batch, n_samples, dim_y); (n_batch, n_sample, dim_x)

        # Denormalize the ground truth
        Y_calib = y_scaler.inverse_transform(Y_calib)
        Y_test = y_scaler.inverse_transform(Y_test)

        # Save the results
        np.save(os.path.join(args.output_saving_path, 'calib_samples.npy'), calib_samples)
        np.save(os.path.join(args.output_saving_path, 'calib_conditions.npy'), calib_conditions)
        np.save(os.path.join(args.output_saving_path, 'test_samples.npy'), test_samples)
        np.save(os.path.join(args.output_saving_path, 'test_conditions.npy'), test_conditions)
        np.save(os.path.join(args.output_saving_path, 'Y_calib.npy'), Y_calib)
        np.save(os.path.join(args.output_saving_path, 'Y_test.npy'), Y_test)

        # Name convention change
        Y_ens_calib, Y_calib, Y_ens_test, Y_test = calib_samples, Y_calib, test_samples, Y_test
        # calib_samples: (n_batch, n_samples, dim_y)
        # Y_calib: (n_batch, dim_y)
    else:

        Y_ens_calib, Y_calib, Y_ens_test, Y_test = get_togo_dataset(dataset_name, data_path=args.data_path)


    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)

    # CP4Gen
    k_hat_list = np.arange(1, args.n_samples + 1)
    qt_list = []
    coverage_list = []
    volume_list = []
    for i in range(len(k_hat_list)):
        k_hat = k_hat_list[i]
        calib_scores = KMeans.summary_score_KMeans(Y_ens_calib, Y_calib, k_hat=k_hat)
        qt = np.quantile(calib_scores, args.coverage) 
        test_scores, test_volumes = KMeans.summary_inference_KMeans(Y_ens_test, Y_test, k_hat=k_hat, qt=qt)

        qt_list.append(qt)
        coverage_list.append(np.mean(test_scores < qt))
        volume_list.append(np.mean(test_volumes))
    
    idx = np.argmin(volume_list)

    print('CP4Gen:', flush=True)
    print(f'k_hat: {k_hat_list[idx]}', flush=True)
    print(f'Empirical coverage: {coverage_list[idx]:.3f}', flush=True)
    print(f'Empirical efficiency: {volume_list[idx]:.3f}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)

    # PCP-VCR
    Y_hat = np.concatenate((Y_ens_calib, Y_ens_test), axis=0)
    Y_cal_test = np.concatenate((Y_calib, Y_test), axis=0).reshape(-1, 1, Y_ens_calib.shape[2])
    # Ranking the samples by their average m-nearest neighbor distances, here we pick m=4.
    # Compute pairwise distances between Y and Y_hat_ranked. Each row is a non-conformity score vector.
    pcp_vcr = PCP.PCP_VCR(n_sample_K = args.n_samples,alpha=0.1,y_dim = Y_ens_calib.shape[2])
    Y_hat_ranked = pcp_vcr.rank(Y_cal_test,Y_hat,k_neighbor = 4)
    dist_matrix = pcp_vcr.compute_dist_matrix(Y_cal_test,Y_hat)
    dist_matrix_rank = pcp_vcr.compute_dist_matrix(Y_cal_test,Y_hat_ranked)
    # Approximate algorithm on calibration data: initialize different entries in range(n_sample), and select the approximated solution with the best approximated efficiency (sum of prediction regions, no consideration of overlap).
    E_q_list = []
    radius_list = []
    for pos in range(args.n_samples):
        E_q = pcp_vcr.calibrate(dist_matrix_rank[:len(calib),:],num_iter = 300,position=pos)
        radius = np.sum(E_q ** pcp_vcr.y_dim)
        E_q_list.append(E_q)
        radius_list.append(radius)
    # Compute the empirical coverage and exact empirical efficiency (with consideration of overlap) on testing data.  
    # get_coverage_length_overlap function is used to compute the exact efficiency of the coverage set, but this is only computable in 1-dim data. For higher dimensions, there is no analytical solution other than Monte Carlo. 
    pcp_vcr_radius = E_q_list[np.argmin(radius_list)]
    emp_coverage = pcp_vcr.empirical_coverage(dist_matrix_rank[len(calib):,:],pcp_vcr_radius)
    rank_pcp_exact_length = PCP.get_coverage_length_overlap(pcp_vcr_radius,Y_hat_ranked[len(calib):])

    print('PCP-VCR:', flush=True)
    print(f'Empirical coverage: {emp_coverage:.3f}', flush=True)
    print(f'Empirical efficiency: {np.mean(rank_pcp_exact_length):.3f}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)

    # PCP.  
    pcp_radius = pcp_vcr.pcp_radius(dist_matrix[:len(calib)])
    pcp_coverage = pcp_vcr.empirical_coverage(dist_matrix[len(calib):],pcp_radius)
    pcp_exact_length = PCP.get_coverage_length_overlap(pcp_radius,Y_hat[len(calib):])

    print('PCP:', flush=True)
    print(f'Empirical coverage: {pcp_coverage:.3f}', flush=True)
    print(f'Empirical efficiency: {np.mean(pcp_exact_length):.3f}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)

    # Save the results
    np.save(os.path.join(args.output_saving_path, 'k_hat_list.npy'), k_hat_list)
    np.save(os.path.join(args.output_saving_path, 'KMeans_coverage.npy'), coverage_list)
    np.save(os.path.join(args.output_saving_path, 'KMeans_volume.npy'), volume_list)
    np.save(os.path.join(args.output_saving_path, 'PCP_coverage.npy'), pcp_coverage)
    np.save(os.path.join(args.output_saving_path, 'PCP_volume.npy'), np.mean(pcp_exact_length))
    np.save(os.path.join(args.output_saving_path, 'PCP_VCR_coverage.npy'), emp_coverage)
    np.save(os.path.join(args.output_saving_path, 'PCP_VCR_volume.npy'), np.mean(rank_pcp_exact_length))



if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--dataset', default='s_curve', type=str)  # dataset name
    parser.add_argument('--data_path', default='./data/', type=str)  # dataset path
    parser.add_argument('--output_saving_path', default='./output/', type=str)  # output saving path

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)

    # PCP parameters
    parser.add_argument('--n_samples', type=int, default=30)
    parser.add_argument('--coverage', type=float, default=0.9)
    #parser.add_argument('--k_hat', type=int, default=3)

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Print configuration
    print('Experiment Configuration:', flush=True)
    print(f'seed: {args.seed}', flush=True)
    print(f'dataset: {args.dataset}', flush=True)
    print(f'data_path: {args.data_path}', flush=True)
    print(f'output_saving_path: {args.output_saving_path}', flush=True)
    print(f'n_epochs: {args.n_epochs}', flush=True)
    print(f'batch_size: {args.batch_size}', flush=True)
    print(f'hidden_dim: {args.hidden_dim}', flush=True)
    print(f'timesteps: {args.timesteps}', flush=True)
    print(f'lr: {args.lr}', flush=True)
    print(f'n_samples: {args.n_samples}', flush=True)
    print(f'coverage: {args.coverage}', flush=True)

    run(args)
