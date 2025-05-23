import os
import sys
import time
import torch
import random
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import PCP
import KMean
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
    # No need to train the generative model for these datasets
    togo_list = ['Mengze_2', 'Mengze_nearest', 'Mengze_3', 'Mengze_4', 'synthetic_normal']

    # Get data set
    if args.dataset not in togo_list:

        if os.path.exists(os.path.join(args.generative_model_path, f'model.pth')):
            Y_ens_calib = np.load(os.path.join(args.generative_model_path, f'Y_ens_calib.npy'))
            Y_calib = np.load(os.path.join(args.generative_model_path, f'Y_calib.npy'))
            Y_ens_test = np.load(os.path.join(args.generative_model_path, f'Y_ens_test.npy'))
            Y_test = np.load(os.path.join(args.generative_model_path, f'Y_test.npy'))        
        else:
            generative_model = GenerativeModel(args)
            generative_model.prep_data()
            generative_model.train()
            Y_ens_calib, calib_conditions, Y_ens_test, test_conditions = generative_model.sample()
            Y_calib, Y_test = generative_model.get_ground_truth()
            generative_model.save()

            np.save(os.path.join(args.generative_model_path, f'Y_ens_calib.npy'), Y_ens_calib)
            np.save(os.path.join(args.generative_model_path, f'Y_calib.npy'), Y_calib)
            np.save(os.path.join(args.generative_model_path, f'Y_ens_test.npy'), Y_ens_test)
            np.save(os.path.join(args.generative_model_path, f'Y_test.npy'), Y_test)
            np.save(os.path.join(args.generative_model_path, f'calib_conditions.npy'), calib_conditions)
            np.save(os.path.join(args.generative_model_path, f'test_conditions.npy'), test_conditions)

    else:
        Y_ens_calib, Y_calib, Y_ens_test, Y_test = get_togo_dataset(args.dataset, data_path=args.data_path)

    # Select a subset of the ensemble size
    ens_size = min(args.n_ens, Y_ens_calib.shape[1])
    Y_ens_calib = Y_ens_calib[:, :ens_size]
    Y_ens_test = Y_ens_test[:, :ens_size]

    print(f'Y_ens_calib shape: {Y_ens_calib.shape}', flush=True)
    print(f'Y_calib shape: {Y_calib.shape}', flush=True)
    print(f'Y_ens_test shape: {Y_ens_test.shape}', flush=True)
    print(f'Y_test shape: {Y_test.shape}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)


    #===========CP4Gen===========
    k_hat_list = [1,2,3,4,5]
    qt_list = []
    coverage_list = []
    volume_list = []
    for i in tqdm(range(len(k_hat_list))):
        k_hat = k_hat_list[i]
        calib_scores = KMean.summary_score_KMeans(Y_ens_calib, Y_calib, k_hat=k_hat)
        qt = np.quantile(calib_scores, args.coverage) 
        test_scores, test_volumes = KMean.summary_inference_KMeans(Y_ens_test, Y_test, k_hat=k_hat, qt=qt)

        #Calculate statistics and save results
        print(f'k_hat: {k_hat}')
        print(f'Test Coverage Rate: {np.mean(test_scores < qt):.6f}')
        print(f'Average Volume: {np.mean(test_volumes):.6f}')
        qt_list.append(qt)
        coverage_list.append(np.mean(test_scores < qt))
        volume_list.append(np.mean(test_volumes))

    idx = np.argmin(volume_list)

    print('CP4Gen:', flush=True)
    print(f'k_hat: {k_hat_list[idx]}', flush=True)
    print(f'Empirical coverage: {coverage_list[idx]:.6f}', flush=True)
    print(f'Empirical efficiency: {volume_list[idx]:.6f}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)


    #===========PCP-VCR===========
    Y_hat = np.concatenate((Y_ens_calib, Y_ens_test), axis=0) # (n_batch, n_samples, dim_y)
    Y_cal_test = np.concatenate((Y_calib, Y_test), axis=0).reshape(-1, 1, Y_ens_calib.shape[2]) # (n_batch, 1, dim_y)
    # Ranking the samples by their average m-nearest neighbor distances, here we pick m=4.
    # Compute pairwise distances between Y and Y_hat_ranked. Each row is a non-conformity score vector.
    pcp_vcr = PCP.PCP_VCR(n_sample_K = args.n_samples,alpha=0.1,y_dim = Y_ens_calib.shape[2])
    dist_matrix = pcp_vcr.compute_dist_matrix(Y_cal_test, Y_hat)
    """
    Y_hat_ranked = pcp_vcr.rank(Y_cal_test,Y_hat,k_neighbor = 4)
    dist_matrix_rank = pcp_vcr.compute_dist_matrix(Y_cal_test,Y_hat_ranked)
    # Approximate algorithm on calibration data: initialize different entries in range(n_sample), and select the approximated solution with the best approximated efficiency (sum of prediction regions, no consideration of overlap).
    E_q_list = []
    radius_list = []
    for pos in tqdm(range(args.n_samples)):
        E_q = pcp_vcr.calibrate(dist_matrix_rank[:len(Y_calib),:],num_iter = 300,position=pos)
        radius = np.sum(E_q ** pcp_vcr.y_dim)
        E_q_list.append(E_q)
        radius_list.append(radius)
    # Compute the empirical coverage and exact empirical efficiency (with consideration of overlap) on testing data.  
    # get_coverage_length_overlap function is used to compute the exact efficiency of the coverage set, but this is only computable in 1-dim data. For higher dimensions, there is no analytical solution other than Monte Carlo. 
    pcp_vcr_radius = E_q_list[np.argmin(radius_list)]
    emp_coverage = pcp_vcr.empirical_coverage(dist_matrix_rank[len(Y_calib):,:],pcp_vcr_radius)
    if Y_ens_calib.shape[2] == 1:
        rank_pcp_exact_length = PCP.get_coverage_length_overlap(pcp_vcr_radius,Y_hat_ranked[len(Y_calib):])
    elif Y_ens_calib.shape[2] == 2:
        rank_pcp_exact_length = PCP.get_coverage_area_overlap_grid(pcp_vcr_radius,Y_hat_ranked[len(Y_calib):])
    else:
        rank_pcp_exact_length = PCP.get_coverage_area_overlap_MC(pcp_vcr_radius,Y_hat_ranked[len(Y_calib):])

    print('PCP-VCR:', flush=True)
    print(f'Empirical coverage: {emp_coverage:.3f}', flush=True)
    print(f'Empirical efficiency: {np.mean(rank_pcp_exact_length):.3f}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)
    """

    #===========PCP===========
    pcp_radius = pcp_vcr.pcp_radius(dist_matrix[:len(Y_calib)])
    pcp_coverage = pcp_vcr.empirical_coverage(dist_matrix[len(Y_calib):],pcp_radius)

    if Y_ens_calib.shape[2] == 1:
        pcp_exact_length = PCP.get_coverage_length_overlap(pcp_radius,Y_hat[len(Y_calib):])
    else:
        pcp_exact_length = PCP.get_coverage_area_overlap(pcp_radius,Y_hat[len(Y_calib):])

    print('PCP:', flush=True)
    print(f'Empirical coverage: {pcp_coverage:.6f}', flush=True)
    print(f'Empirical efficiency: {np.mean(pcp_exact_length):.6f}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)

    #===========Save results===========
    os.makedirs(os.path.join(args.output_saving_path, args.dataset), exist_ok=True)
    # np.save(os.path.join(args.output_saving_path, f'{args.dataset}/k_hat_list.npy'), k_hat_list)
    # np.save(os.path.join(args.output_saving_path, f'{args.dataset}/KMeans_coverage.npy'), coverage_list)
    # np.save(os.path.join(args.output_saving_path, f'{args.dataset}/KMeans_volume.npy'), volume_list)
    np.save(os.path.join(args.output_saving_path, f'{args.dataset}/PCP_coverage.npy'), pcp_coverage)
    np.save(os.path.join(args.output_saving_path, f'{args.dataset}/PCP_volume.npy'), np.mean(pcp_exact_length))
    # np.save(os.path.join(args.output_saving_path, f'{args.dataset}/PCP_VCR_coverage.npy'), emp_coverage)
    # np.save(os.path.join(args.output_saving_path, f'{args.dataset}/PCP_VCR_volume.npy'), np.mean(rank_pcp_exact_length))







if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--dataset', default='s_curve', type=str)  # dataset name
    parser.add_argument('--data_path', default='./data/', type=str)  # dataset path
    parser.add_argument('--output_saving_path', default='./output/', type=str)  # output saving path
    parser.add_argument('--generative_model_path', default='./generative_models/', type=str)  # generative model path

    # Training parameters
    parser.add_argument('--model_type', type=str, default='flow-matching')
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_samples', type=int, default=300)

    # CP parameters
    parser.add_argument('--CP_type', type=str, default='PCP')
    parser.add_argument('--n_ens', type=int, default=30)
    parser.add_argument('--coverage', type=float, default=0.9)
    parser.add_argument('--max_k', type=int, default=10)

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.model_type == 'flow-matching':
        model_params = {
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
            'timesteps': args.timesteps,
            'lr': args.lr,
        }

    model_params = args.model_type + ''.join([f'--{k}={v}' for k, v in model_params.items()])
    args.generative_model_path = os.path.join(args.generative_model_path, model_params)
    args.generative_model_path = os.path.join(args.generative_model_path, args.dataset)

    if not os.path.exists(args.generative_model_path):
        os.system(f'mkdir -p {args.generative_model_path}')

    # Print configuration
    print('Experiment Configuration:', flush=True)
    print(f'seed: {args.seed}', flush=True)
    print(f'dataset: {args.dataset}', flush=True)
    print(f'data_path: {args.data_path}', flush=True)
    print(f'output_saving_path: {args.output_saving_path}', flush=True)
    print(f'generative_model_path: {args.generative_model_path}', flush=True)
    print(f'model_type: {args.model_type}', flush=True)
    print(f'n_epochs: {args.n_epochs}', flush=True)
    print(f'batch_size: {args.batch_size}', flush=True)
    print(f'hidden_dim: {args.hidden_dim}', flush=True)
    print(f'timesteps: {args.timesteps}', flush=True)
    print(f'lr: {args.lr}', flush=True)
    print(f'n_samples: {args.n_samples}', flush=True)
    print(f'CP_type: {args.CP_type}', flush=True)
    print(f'n_ens: {args.n_ens}', flush=True)
    print(f'coverage: {args.coverage}', flush=True)
    print(f'max_k: {args.max_k}', flush=True)

    # Run the experiment
    run(args)
