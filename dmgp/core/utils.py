import sys
sys.path.append('../../') # Add DMGP root

import os
import os.path as osp
from tqdm import tqdm
import yaml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from numpy.random import default_rng
rng = default_rng() # Add seed if needed

import matplotlib.pyplot as plt
plt.style.use("ggplot")

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import gpflow
from gpflow.kernels import SquaredExponential
from gpflow.inducing_variables import InducingPoints
from gpflow.likelihoods import Gaussian

import dmgp

# Plots the ELBO
def plot_elbo(elbos, log_dir, show_plot=False):
    if tf.is_tensor(elbos[0]):
        elbos = [elbo.numpy() for elbo in elbos]

    n_epochs = len(elbos)

    plt.figure(figsize=(8,6))
    plt.plot(list(range(1,n_epochs+1)), elbos)
    plt.title('ELBO')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    path = osp.join(log_dir, 'elbo_plot.png')
    plt.savefig(path)
    print(f'ELBO plot generated: {path}')
    
    if show_plot:
        plt.show()

# Runs K-means on input points for inducing input initialization
def ind_kmeans(m, data):
    kmeans = KMeans(n_clusters=m).fit(data)
    ind_points = kmeans.cluster_centers_

    return ind_points

# Callback function for printing average ELBO
def print_avg_elbo(epoch_id=None, elbo=None, exp=None, prior_kl=None, **_):
    tf.print(f'[Epoch {epoch_id}] ELBO: {elbo:.4f}\tPrior KL: {prior_kl:.4f}')

# Callback function for relaying average ELBO
def avg_elbo_cb(elbo=None, **_):
    return elbo

# Callback function for relaying KL
def kl_cb(kl=None, **_):
    return kl

# Callback function for relaying average expectation
def exp_cb(exp=None, **_):
    return exp

# Callback function for relaying elapsed time at each epoch
def time_cb(time=None, **_):
    return time