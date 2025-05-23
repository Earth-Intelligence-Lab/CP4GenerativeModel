import os
import sys
import time
import torch
import random
import argparse
import flow_matching
import numpy as np

from dataset import *
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


class GenerativeModel:
    def __init__(self, args):
        self.args = args
        self.model_type = args.model_type
        
        self.X, self.Y = get_dataset(self.args.dataset, data_path=self.args.data_path)

        if self.model_type == 'flow-matching':
            self.model = flow_matching.FlowMatchingNet(input_dim=self.Y.shape[1], condition_dim=self.X.shape[1], hidden_dim=self.args.hidden_dim).to(self.args.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

            linear_alpha = flow_matching.LinearAlpha()
            squareroot_beta = flow_matching.SquareRootBeta()
            self.gaussian_path = flow_matching.GaussianPath(linear_alpha, squareroot_beta)

    def prep_data(self):
        N = self.X.shape[0]

        train, calib, test = np.split(range(N), [int(.6 * N), int(.8 * N), ])
        print(f'train size: {len(train)}, calib size: {len(calib)}, test size: {len(test)}', flush=True)

        # Extract train, calib, and test subsets
        X_train, Y_train = self.X[train], self.Y[train]
        X_calib, Y_calib = self.X[calib], self.Y[calib]
        X_test, Y_test = self.X[test], self.Y[test]

        # Standardize the data
        self.x_scaler = StandardScaler()
        X_train = self.x_scaler.fit_transform(X_train)
        X_calib = self.x_scaler.transform(X_calib)
        X_test = self.x_scaler.transform(X_test)

        self.y_scaler = StandardScaler()
        Y_train = self.y_scaler.fit_transform(Y_train)
        Y_calib = self.y_scaler.transform(Y_calib)
        Y_test = self.y_scaler.transform(Y_test)

        # Create Dataset objects
        train_dataset = DatasetTensor(X_train, Y_train)
        calib_dataset = DatasetTensor(X_calib, Y_calib)
        test_dataset = DatasetTensor(X_test, Y_test)

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.calib_loader = DataLoader(calib_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
    
    def train(self):
        if self.model_type == 'flow-matching':
            flow_matching.train_flow_matching(self.model, self.gaussian_path, self.train_loader, self.optimizer, self.args.n_epochs, self.args.device)

    def sample(self):
        if self.model_type == 'flow-matching':
            calib_samples, calib_conditions = flow_matching.generate_samples_for_dataset(self.model, self.gaussian_path, self.calib_loader, self.args.n_samples, self.args.timesteps, self.args.device)
            test_samples, test_conditions = flow_matching.generate_samples_for_dataset(self.model, self.gaussian_path, self.test_loader, self.args.n_samples, self.args.timesteps, self.args.device)

        # Denormalize the samples and conditions
        calib_samples = self.y_scaler.inverse_transform(calib_samples.reshape(-1, calib_samples.shape[-1])).reshape(calib_samples.shape)
        calib_conditions = self.x_scaler.inverse_transform(calib_conditions.reshape(-1, calib_conditions.shape[-1])).reshape(calib_conditions.shape)
        test_samples = self.y_scaler.inverse_transform(test_samples.reshape(-1, test_samples.shape[-1])).reshape(test_samples.shape)
        test_conditions = self.x_scaler.inverse_transform(test_conditions.reshape(-1, test_conditions.shape[-1])).reshape(test_conditions.shape)
        # (n_batch, n_samples, dim_y); (n_batch, n_sample, dim_x)        

        return calib_samples, calib_conditions, test_samples, test_conditions
    
    def get_ground_truth(self):
        N = self.X.shape[0]
        train, calib, test = np.split(range(N), [int(.6 * N), int(.8 * N), ])

        Y_calib = self.Y[calib]
        Y_test = self.Y[test]

        return Y_calib, Y_test

    def save(self):
        filename = os.path.join(self.args.generative_model_path, f'model.pth')
        torch.save(self.model.state_dict(), filename)

    def load(self):
        filename = os.path.join(self.args.generative_model_path, f'model.pth')
        self.model.load_state_dict(torch.load(filename))











        