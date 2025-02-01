import os
import sys
import torch
import random
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

""" Load dataset methods """
from dataset import generate_s_curve
from sklearn.preprocessing import StandardScaler
import time

""" Load generative model methods """
from CP4GenerativeModel.functions import summary_score, summary_inference
from models.test import generate_mix_Gaussian
import multiprocessing


def main(args):
    n_exp = args.n_exp

    for exp in tqdm(range(n_exp)):

        """
        prepare the data
        """

        """
        X, Y = generate_s_curve(n_total)
        n_test = min(2000, int(n_total * 0.2))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=n_test)
        n_cal = min(2000, int(X_train.shape[0] * 0.1))
        X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, test_size=n_cal)


        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_calib = scaler.transform(X_calib)
        X_test = scaler.transform(X_test)

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        assert(n_cal == X_calib.shape[0])


        if len(X.shape) == 1:
            n_features = 1
        else:
            n_features = X.shape[1]
        """


        """
        Train Probabilistic Model: collect ensembles on calibration set
        """
        # model = ...
        # model.fit(X_train, Y_train)
        # model.predict(X_test)
        # currently we use test.py to generate ensembles
        n_calib = 10000; n_test = 2000; n_ens = 100
        
        

        """
        Conformal Prediction: compute quantiles
        """
        _, calib_data = generate_mix_Gaussian(n_calib, n_ens, k=3, d=2)
        calib_scores = summary_score(calib_data, k_hat=3)
        qt = np.quantile(calib_scores, 0.9)

        """
        Validate on test set: get coverage and average volume
        """
        _, test_data = generate_mix_Gaussian(n_test, n_ens, k=3, d=2)
        k_hat = 3
        test_scores, test_volumes = summary_inference(test_data, k_hat, qt=qt, grid_res=200, buffle=3)


        """
        Calculate statistics and save results
        """

        print(f'Epoch {exp}:-------------------')
        print(f'k_hat: {k_hat}')
        print(f'Test Coverage Rate: {np.mean(test_scores < qt):.2f}')
        print(f'Average Volume: {np.mean(test_volumes):.2f}')

    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_exp", type=int, default=5)
    args = parser.parse_args()
    main(args)

