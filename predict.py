import torch
import argparse

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

""" Load dataset methods """

""" Load generative model methods """
from KMeans import summary_score, summary_inference


def main(args):
    """
    load data
    """

    X_calib, X_test = np.load(f'data/{args.dataset}/calib_X.npy'), np.load(f'data/{args.dataset}/test_X.npy')
    Y_calib, Y_test = np.load(f'data/{args.dataset}/calib_Y.npy'), np.load(f'data/{args.dataset}/test_Y.npy')
    Y_hat_calib, Y_hat_test = np.load(f'data/{args.dataset}/calib_Y_hat.npy'), np.load(f'data/{args.dataset}/test_Y_hat.npy')
    
    """
    Conformal Prediction: compute quantiles
    """
    # _, calib_data = generate_mix_Gaussian(n_calib, n_samples, k=3, d=2)
    print(f'Y_calib shape: {Y_calib.shape}')
    print(f'Y_hat_calib shape: {Y_hat_calib.shape}')
    calib_scores = summary_score(Y_calib, Y_hat_calib, k_hat=3)
    qt = np.quantile(calib_scores, 0.9)

    """
    Validate on test set: get coverage and average volume
    """
    k_hat = 3
    test_scores, test_volumes = summary_inference(Y_test, Y_hat_test, k_hat, qt=qt, grid_res=200, buffle=3)


    """
    Calculate statistics and save results
    """

    print(f'k_hat: {k_hat}')
    print(f'Test Coverage Rate: {np.mean(test_scores < qt):.2f}')
    print(f'Average Volume: {np.mean(test_volumes):.2f}')

    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='s_curve', type=str)  # dataset name
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1000)
    # PCP parameters
    parser.add_argument('--n_samples', type=int, default=10)

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main(args)

