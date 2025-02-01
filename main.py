import os
import sys
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from flow_matching import *
from torch.utils.data import DataLoader


def run(args):

    dataset_name = args.dataset
    print(f'data: {dataset_name}')

    X, Y = get_dataset(dataset_name, base_path=None)
    N = X.shape[0]

    train, calib, test = np.split(range(N), [int(.6 * N), int(.8 * N), ])
    print(f'train size: {len(train)}, calib size: {len(calib)}, test size: {len(test)}')

    # Extract train, calib, and test subsets
    X_train, Y_train = X[train], Y[train]
    X_calib, Y_calib = X[calib], Y[calib]
    X_test, Y_test = X[test], Y[test]

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

    # Validate Results
    plt.scatter(
        np.vstack(calib_conditions),
        np.vstack(calib_samples),
        alpha=0.2, s=3, label='Random Samples', color='red'
    )

    plt.scatter(
        X_calib,
        Y_calib,
        alpha=0.5, s=10, label='Truth', color='blue'
    )

    plt.legend(loc='upper right')


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='s_curve', type=str)  # dataset name
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    # PCP parameters
    parser.add_argument('--n_samples', type=int, default=10)

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    run(args)
