import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture


class CPGen:
    def __init__(self, k):