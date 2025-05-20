import torch
import numpy as np
import pandas as pd

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


def generate_synthetic_generative_data(N, N_ens, k, d, seed=100):
    # Using normal distribution to generate synthetic data
    # N: size of dataset
    # N_ens: size of ensemble set
    # k: max possible gaussian components
    # d: dimension of gaussian variables

    # data: (N, N_ens, d)

    np.random.seed(seed)

    W = np.random.uniform(0, np.sqrt(k), (N, k))
    W = np.exp(W) / np.sum(np.exp(W), axis=1, keepdims=True) # (N, k)

    Mu = np.random.uniform(-5, 5, (N, k, d)) # (N, k, d)

    M = np.random.rand(N * k, d, d) -.5 + np.tile(np.eye(d), (N * k, 1, 1))
    M_T = np.transpose(M, axes=(0, 2, 1))

    Sigma = np.matmul(M, M_T)
    Sigma = Sigma.reshape((N, k, d, d)) # (N, k, d, d)

    guassian_index = []
    for i in range(N):
        guassian_index.append(np.random.choice(k, size=N_ens, replace=True, p=W[i]))
    guassian_index = np.array(guassian_index) # (N, N_ens)

    data = []
    for i in range(N):
        ens = []
        for j in range(N_ens):
            mu = Mu[i][guassian_index[i][j]]
            sigma = Sigma[i][guassian_index[i][j]]
            ens.append(np.random.multivariate_normal(mu, sigma))
        data.append(ens) # (N, N_ens, d)

    return np.array(data)


def get_togo_dataset(name, data_path=None):

    # name: dataset name
    # data_path: the path to datasets

    # Y_ens_calib: (n_calib, n_samples, dim_y)
    # Y_calib: (n_calib, dim_y)
    # Y_ens_test: (n_test, n_samples, dim_y)
    # Y_test: (n_test, dim_y)

    if name == 'Mengze_2':
        Y_ens_calib = np.load(data_path + 'Mengze/y_hat2_reduce_NSM.npy')
        Y_calib = np.load(data_path + 'Mengze/y2_reduce_NSM.npy')[:, 0, :]

        Y_ens_test = np.load(data_path + 'Mengze/y_hat1_reduce_NSM.npy')
        Y_test = np.load(data_path + 'Mengze/y1_reduce_NSM.npy')[:, 0, :]

    if name == 'Mengze_3':
        Y_ens_calib = np.load(data_path + 'Mengze/y_hat2_reduce_NMSM.npy')
        Y_calib = np.load(data_path + 'Mengze/y2_reduce_NMSM.npy')[:, 0, :]

        Y_ens_test = np.load(data_path + 'Mengze/y_hat1_reduce_NMSM.npy')
        Y_test = np.load(data_path + 'Mengze/y1_reduce_NMSM.npy')[:, 0, :]

    if name == 'Mengze_4':
        Y_ens_calib = np.load(data_path + 'Mengze/y_hat2_reduce_QM.npy')
        Y_calib = np.load(data_path + 'Mengze/y2_reduce_QM.npy')[:, 0, :]

        Y_ens_test = np.load(data_path + 'Mengze/y_hat1_reduce_QM.npy')
        Y_test = np.load(data_path + 'Mengze/y1_reduce_QM.npy')[:, 0, :]

    if name == 'synthetic_normal':
        data = generate_synthetic_generative_data(N=2000, N_ens=41, k=3, d=2, seed=100)
        # (N, N_ens, d)

        Y_ens_calib = data[:1000, :40, :]
        Y_calib = data[:1000, 40, :]

        Y_ens_test = data[1000:, :40, :]
        Y_test = data[1000:, 40, :]

    return Y_ens_calib, Y_calib, Y_ens_test, Y_test


def get_dataset(name, data_path=None, seed=0):

    # name: dataset name
    # data_path: the path to datasets

    # X: (N, dim_x)
    # Y: (N, dim_y)

    # Set random seed
    np.random.seed(seed)

    if name == 's_curve':
        n = 5000
        x, t = make_s_curve(n_samples=n, noise=0, random_state=seed)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, 2].reshape(n, 1)

    if name == 'spiral':
        n = 5000
        x, t = make_swiss_roll(n_samples=n, noise=0, random_state=seed)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == 'circle':
        n = 5000
        x, y = make_circles(n_samples=n, noise=0, factor=0.7, random_state=seed)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == 'moon':
        n = 5000
        x, y = make_moons(n_samples=n, noise=0.01, random_state=seed)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == '25-Gaussians':
        n = 5000
        x_coord, y_coord = np.meshgrid(np.linspace(-1.5, 1.5, 5), np.linspace(-1.5, 1.5, 5))
        coords = np.concatenate([x_coord.reshape(-1, 1), y_coord.reshape(-1, 1)], axis=1)
        x, y = make_blobs(n_samples=n, centers=coords, cluster_std=0.01, random_state=seed)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == '8-Gaussians':
        n = 5000
        rad = np.linspace(-1, 1, 9)[:-1] * np.pi
        x_coord, y_coord = np.cos(rad) * 1.5, np.sin(rad) * 1.5
        coords = np.concatenate([x_coord.reshape(-1, 1), y_coord.reshape(-1, 1)], axis=1)
        x, y = make_blobs(n_samples=n, centers=coords, cluster_std=0.01, random_state=seed)

        X = x[:, 0].reshape(n, 1)
        Y = x[:, -1].reshape(n, 1)

    if name == "meps_19":
        df = pd.read_csv(data_path + 'meps_19_reg.csv')

        N = len(df)
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df[col_names].values.reshape(N, -1)[idx]
        Y = df[[response_name]].values.reshape(N, -1)[idx]

    if name == "meps_20":
        df = pd.read_csv(data_path + 'meps_20_reg.csv')

        N = len(df)
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df[col_names].values.reshape(N, -1)[idx]
        Y = df[[response_name]].values.reshape(N, -1)[idx]

    if name == "meps_21":
        df = pd.read_csv(data_path + 'meps_21_reg.csv')

        N = len(df)
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df[col_names].values.reshape(N, -1)[idx]
        Y = df[[response_name]].values.reshape(N, -1)[idx]

    if name == "facebook_1":
        df = pd.read_csv(data_path + 'facebook_1.csv')

        N = len(df)
        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df.iloc[:, 0:53].values.reshape(N, -1)[idx]
        Y = df.iloc[:, [53]].values.reshape(N, -1)[idx]

    if name == "facebook_2":
        df = pd.read_csv(data_path + 'facebook_2.csv')

        N = len(df)
        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df.iloc[:, 0:53].values.reshape(N, -1)[idx]
        Y = df.iloc[:, [53]].values.reshape(N, -1)[idx]

    if name == "bio":
        df = pd.read_csv(data_path + 'CASP.csv')

        N = len(df)
        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df.iloc[:, 1:].values.reshape(N, -1)[idx]
        Y = df.iloc[:, [0]].values.reshape(N, -1)[idx]

    if name == 'blog_data':
        df = pd.read_csv(data_path + 'blogData_train.csv', header=None)

        N = len(df)
        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df.iloc[:, 0:280].values.reshape(N, -1)[idx]
        Y = df.iloc[:, [-1]].values.reshape(N, -1)[idx]

    if name == "temperature":
        df = pd.read_csv(data_path + 'Bias_correction_ucl.csv')
        df = df.drop(columns=['station', 'Date', 'Next_Tmax'])
        df = df.dropna()

        N = len(df)
        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df.iloc[:, :-1].values.reshape(N, -1)[idx]
        Y = df.iloc[:, [-1]].values.reshape(N, -1)[idx]

    if name == "bike":
        df = pd.read_csv(data_path + 'bike_train.csv')

        # # seperating season as per values. this is bcoz this will enhance features.
        season = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season], axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather = pd.get_dummies(df['weather'], prefix='weather')
        df = pd.concat([df, weather], axis=1)

        # # # now can drop weather and season.
        df.drop(['season', 'weather'], inplace=True, axis=1)

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011: 0, 2012: 1})

        df.drop('datetime', axis=1, inplace=True)
        df.drop(['casual', 'registered'], axis=1, inplace=True)

        N = len(df)
        # Shuffle dataset
        idx = np.arange(N)
        np.random.shuffle(idx)

        X = df.drop('count', axis=1).values.reshape(N, -1)[idx]
        Y = df[['count']].values.reshape(N, -1)[idx]

    return X, Y