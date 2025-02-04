import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs


class DatasetTensor(Dataset):
    def __init__(self, X, Y):
        """
        Args:
            X: Input data (features), shape (N, dim_x).
            Y: Target data (labels), shape (N, dim_y).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_dataset(name, base_path=None):

    # name: dataset name
    # base_path: the path to datasets

    # X: (N, dim_x)
    # Y: (N, dim_y)

    if name == 's_curve':
        n = 5000
        x, t = make_s_curve(n_samples=n, noise=0)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, 2].reshape(n, 1)

    if name == 'spiral':
        n = 5000
        x, t = make_swiss_roll(n_samples=n, noise=0)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == 'circle':
        n = 5000
        x, y = make_circles(n_samples=n, noise=0, factor=0.7)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == 'moon':
        n = 5000
        x, y = make_moons(n_samples=n, noise=0.01)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == '25-Gaussians':
        n = 5000
        x_coord, y_coord = np.meshgrid(np.linspace(-1.5, 1.5, 5), np.linspace(-1.5, 1.5, 5))
        coords = np.concatenate([x_coord.reshape(-1, 1), y_coord.reshape(-1, 1)], axis=1)
        x, y = make_blobs(n_samples=n, centers=coords, cluster_std=0.01)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == '8-Gaussians':
        n = 5000
        rad = np.linspace(-1, 1, 9)[:-1] * np.pi
        x_coord, y_coord = np.cos(rad) * 1.5, np.sin(rad) * 1.5
        coords = np.concatenate([x_coord.reshape(-1, 1), y_coord.reshape(-1, 1)], axis=1)
        x, y = make_blobs(n_samples=n, centers=coords, cluster_std=0.01)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    return X, Y
