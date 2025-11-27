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
import configurations
from dataset import *
from utility import *
from flow_matching import *
from CP_generative import *
from generative_models import *
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

        if os.path.exists(os.path.join(args.model_path, f'model.pth')):
            Y_ens_calib = np.load(os.path.join(args.model_path, f'Y_ens_calib.npy'))
            calib_scores = np.load(os.path.join(args.model_path, f'calib_scores.npy'))
            Y_calib = np.load(os.path.join(args.model_path, f'Y_calib.npy'))
            Y_ens_test = np.load(os.path.join(args.model_path, f'Y_ens_test.npy'))
            test_scores = np.load(os.path.join(args.model_path, f'test_scores.npy'))
            Y_test = np.load(os.path.join(args.model_path, f'Y_test.npy'))

        else:
            generative_model = GenerativeModel(args)
            generative_model.prep_data()
            generative_model.train()
            Y_ens_calib, calib_scores, calib_conditions, Y_ens_test, test_scores, test_conditions = generative_model.sample()
            Y_calib, Y_test = generative_model.get_ground_truth()
            generative_model.save()

            np.save(os.path.join(args.model_path, f'Y_ens_calib.npy'), Y_ens_calib)
            np.save(os.path.join(args.model_path, f'calib_scores.npy'), calib_scores)
            np.save(os.path.join(args.model_path, f'Y_calib.npy'), Y_calib)
            np.save(os.path.join(args.model_path, f'Y_ens_test.npy'), Y_ens_test)
            np.save(os.path.join(args.model_path, f'test_scores.npy'), test_scores)
            np.save(os.path.join(args.model_path, f'Y_test.npy'), Y_test)
            np.save(os.path.join(args.model_path, f'calib_conditions.npy'), calib_conditions)
            np.save(os.path.join(args.model_path, f'test_conditions.npy'), test_conditions)

    else:
        Y_ens_calib, Y_calib, Y_ens_test, Y_test = get_togo_dataset(args.dataset, data_path=args.data_path)

    # Select a subset of the ensemble size
    ens_size = min(args.n_ens, Y_ens_calib.shape[1])
    Y_ens_calib = Y_ens_calib[:, :ens_size]
    Y_ens_test = Y_ens_test[:, :ens_size]
    calib_scores = calib_scores[:, :ens_size]
    test_scores = test_scores[:, :ens_size]

    print(f'Y_ens_calib shape: {Y_ens_calib.shape}', flush=True)
    print(f'calib_scores shape: {calib_scores.shape}', flush=True)
    print(f'Y_calib shape: {Y_calib.shape}', flush=True)
    print(f'Y_ens_test shape: {Y_ens_test.shape}', flush=True)
    print(f'test_scores shape: {test_scores.shape}', flush=True)
    print(f'Y_test shape: {Y_test.shape}', flush=True)

    print(' ', flush=True)
    print('--------------------------------', flush=True)
    print(' ', flush=True)

    #Select Conformal Prediction Method
    if args.CP_type == 'PCP':
        cp_method = CPGen(args, k=ens_size)
        cp_method.fit(Y_ens_calib, Y_calib)
        scores, volumes, ks = cp_method.predict(Y_ens_test, Y_test)

        print(f'Test Coverage Rate: {np.mean(scores < cp_method.quant_score):.6f}', flush=True)
        print(f'Average k: {np.mean(ks):.6f}', flush=True)
        print(f'Average Volume: {np.mean(volumes):.6f}', flush=True)

        np.save(os.path.join(args.output_saving_path, f'PCP_scores.npy'), scores)
        np.save(os.path.join(args.output_saving_path, f'PCP_volumes.npy'), volumes)
        np.save(os.path.join(args.output_saving_path, f'PCP_ks.npy'), ks)
        np.save(os.path.join(args.output_saving_path, f'PCP_quant_score.npy'), np.array([cp_method.quant_score]))

    if args.CP_type == 'HD-PCP':
        keep_rates = [1, 0.95, 0.6, 0.5, 0.3]

        for keep_rate in keep_rates:
            ens_size_keep = int(ens_size * keep_rate)
            cp_method = CPGen(args, k=ens_size_keep)

            # Select sample with top scores
            topk_idx_calib = np.argsort(calib_scores.squeeze(-1), axis=1)[:, -ens_size_keep:]
            topk_idx_test = np.argsort(test_scores.squeeze(-1), axis=1)[:, -ens_size_keep:]

            N_calib = Y_ens_calib.shape[0]
            N_test = Y_ens_test.shape[0]

            cp_method.fit(Y_ens_calib[np.arange(N_calib)[:, None], topk_idx_calib, :], Y_calib)
            scores, volumes, ks = cp_method.predict(Y_ens_test[np.arange(N_test)[:, None], topk_idx_test, :], Y_test)

            print(f'Keep Ensemble Size: {ens_size_keep}', flush=True)
            print(f'Test Coverage Rate: {np.mean(scores < cp_method.quant_score):.6f}', flush=True)
            print(f'Average k: {np.mean(ks):.6f}', flush=True)
            print(f'Average Volume: {np.mean(volumes):.6f}', flush=True)
            print(' ', flush=True)

            np.save(os.path.join(args.output_saving_path, f'HD-PCP_scores_{keep_rate}.npy'), scores)
            np.save(os.path.join(args.output_saving_path, f'HD-PCP_volumes_{keep_rate}.npy'), volumes)
            np.save(os.path.join(args.output_saving_path, f'HD-PCP_ks_{keep_rate}.npy'), ks)
            np.save(os.path.join(args.output_saving_path, f'HD-PCP_quant_score_{keep_rate}.npy'), np.array([cp_method.quant_score]))

    if args.CP_type == 'CP4Gen':
        d = Y_ens_calib.shape[-1]
        k_list = get_k_list(ens_size, d)

        for k in k_list:
            cp_method = CPGen(args, k=k)
            cp_method.fit(Y_ens_calib, Y_calib)
            scores, volumes, ks = cp_method.predict(Y_ens_test, Y_test)

            print(f'K: {k}', flush=True)
            print(f'Test Coverage Rate: {np.mean(scores < cp_method.quant_score):.6f}', flush=True)
            print(f'Average k: {np.mean(ks):.6f}', flush=True)
            print(f'Average Volume: {np.mean(volumes):.6f}', flush=True)
            print(' ', flush=True)

            np.save(os.path.join(args.output_saving_path, f'CP4Gen_scores_{k}.npy'), scores)
            np.save(os.path.join(args.output_saving_path, f'CP4Gen_volumes_{k}.npy'), volumes)
            np.save(os.path.join(args.output_saving_path, f'CP4Gen_ks_{k}.npy'), ks)
            np.save(os.path.join(args.output_saving_path, f'CP4Gen_quant_score_{k}.npy'), np.array([cp_method.quant_score]))

    if args.CP_type == 'CP4Gen_Adaptive':
        w_thred_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]

        for w_thred in w_thred_list:
            cp_method = CPGen_Adaptive(args, w_thred=w_thred)
            cp_method.fit(Y_ens_calib, Y_calib)
            scores, volumes, ks = cp_method.predict(Y_ens_test, Y_test)

            print(f'Weight Threshold: {w_thred}', flush=True)
            print(f'Test Coverage Rate: {np.mean(scores < cp_method.quant_score):.6f}', flush=True)
            print(f'Average k: {np.mean(ks):.6f}', flush=True)
            print(f'Average Volume: {np.mean(volumes):.6f}', flush=True)
            print(' ', flush=True)

            np.save(os.path.join(args.output_saving_path, f'CP4Gen_Adaptive_scores_{w_thred}.npy'), scores)
            np.save(os.path.join(args.output_saving_path, f'CP4Gen_Adaptive_volumes_{w_thred}.npy'), volumes)
            np.save(os.path.join(args.output_saving_path, f'CP4Gen_Adaptive_ks_{w_thred}.npy'), ks)
            np.save(os.path.join(args.output_saving_path, f'CP4Gen_Adaptive_quant_score_{w_thred}.npy'), np.array([cp_method.quant_score]))


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--dataset', default='s_curve', type=str)  # dataset name
    parser.add_argument('--data_path', default='./data/', type=str)  # dataset path
    parser.add_argument('--output_saving_path', default='./output/', type=str)  # output saving path
    parser.add_argument('--model_path', default='./generative_models/', type=str)  # generative model path

    # Training parameters
    parser.add_argument('--model_type', type=str, default='flow-matching')
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_samples', type=int, default=1000)

    # CP parameters
    parser.add_argument('--CP_type', type=str, default='PCP')
    parser.add_argument('--n_ens', type=int, default=30)
    parser.add_argument('--coverage', type=float, default=0.9)
    parser.add_argument('--max_k', type=int, default=10)

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.model_type in ['flow-matching-HD', 'diffusion', 'diffusion_sparse']:
        model_params = {
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
            'timesteps': args.timesteps,
            'lr': args.lr,
        }

    model_params = args.model_type + ''.join([f'--{k}={v}' for k, v in model_params.items()])
    args.model_path = os.path.join(args.model_path, model_params)
    args.model_path = os.path.join(args.model_path, args.dataset)

    if not os.path.exists(args.model_path):
        os.system(f'mkdir -p {args.model_path}')

    # Print configuration
    print('Experiment Configuration:', flush=True)
    print(f'seed: {args.seed}', flush=True)
    print(f'dataset: {args.dataset}', flush=True)
    print(f'data_path: {args.data_path}', flush=True)
    print(f'output_saving_path: {args.output_saving_path}', flush=True)
    print(f'model_path: {args.model_path}', flush=True)
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
