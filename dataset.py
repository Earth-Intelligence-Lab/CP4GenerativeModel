import numpy as np

from torch.utils.data import Dataset
from sklearn.datasets import make_s_curve


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

    return X, Y
