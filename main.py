#Import Modules
import os
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np

import matplotlib as plt


print('Tensorflow Version: ', tf.__version__)


def firm(K, eta, alpha, delta):
    """Calculate return, wage and aggregate production.

    r = eta * K^(alpha-1) * L^(1-alpha) + (1-delta)
    w = eta * K^(alpha) * L^(-alpha)
    Y = eta * K^(alpha) * L^(1-alpha) + (1-delta) * K

    Args:
        K: aggregate capital,
        eta: TFP value,
        alpha: output elasticity,
        delta: depreciation value.

    Returns:
        return: return (marginal product of capital),
        wage: wage (marginal product of labor),
        Y: aggregate production.
    """
    L = tf.ones_like(K)

    r = alpha * eta * K ** (alpha - 1) * L ** (1 - alpha) + (1 - delta)
    w = (1 - alpha) * eta * K ** alpha * L ** (-alpha)
    Y = eta * K ** alpha * L ** (1 - alpha) + (1 - delta) * K

    return r, w, Y


def shocks(z, eta, delta):
    """Calculates tfp and depreciation based on current exogenous shock.

    Args:
        z: current exogenous shock (in {1, 2, 3, 4}),
        eta: tensor of TFP values to sample from,
        delta: tensor of depreciation values to sample from.

    Returns:
        tfp: TFP value of exogenous shock,
        depreciation: depreciation values of exogenous shock.
    """

    tfp = tf.gather(eta, tf.cast(z, tf.int32))
    depreciation = tf.gather(delta, tf.cast(z, tf.int32))
    return tfp, depreciation


def wealth(k, R, l, W):
    """Calculates the wealth of the agents.

    Args:
        k: capital distribution,
        R: matrix of return,
        l: labor distribution,
        W: matrix of wages.

    Returns:
        fin_wealth: financial wealth distribution,
        lab_wealth: labor income distribution,
        tot_income: total income distribution.
    """

    fin_wealth = k * R
    lab_wealth = l * W
    tot_income = tf.add(fin_wealth, lab_wealth)
    return fin_wealth, lab_wealth, tot_income




def main():
    seed = 1
    num_hidden_nodes = [100, 100]
    activations_hidden_nodes = [tf.nn.relu, tf.nn.relu]
    optimizer = 'adam'
    num_episodes = 50000
    len_episodes = 10240
    epochs_per_episode = 20
    minibatch_size = 512
    num_minibatch = int(len_episodes/minibatch_size)
    lr = 1e-5






