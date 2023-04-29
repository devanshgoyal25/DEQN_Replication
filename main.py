#Import Modules
import os
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np

import matplotlib as plt


print('Tensorflow Version: ', tf.__version__)

seed = 1
num_hidden_nodes = [100, 50]
activations_hidden_nodes = [tf.nn.relu, tf.nn.relu]
optimizer = 'adam'
num_episodes = 50000
len_episodes = 10240
epochs_per_episode = 20
minibatch_size = 512
num_minibatch = int(len_episodes/minibatch_size)
lr = 1e-5


num_agents = A = 6
num_exogenous_shocks = 4

# Number of agents and exogenous shocks
num_agents = A = 6
num_exogenous_shocks = 4

# Exogenous shock values
delta = tf.constant([[0.5], [0.5], [0.9], [0.9]],  dtype=tf.float32)  # Capital depreciation (dependent on shock)
eta = tf.constant([[0.95], [1.05], [0.95], [1.05]], dtype=tf.float32)  # TFP shock (dependent on shock)

# Transition matrix
# In this example we hardcoded the transition matrix. Changes cannot be made without also changing
# the corresponding code below.
p_transition = 0.25  # All transition probabilities are 0.25
pi_np = p_transition * np.ones((4, 4))  # Transition probabilities
pi = tf.constant(pi_np, dtype=tf.float32)  # Transition probabilities

# Labor endowment
labor_endow_np = np.zeros((1, A))
labor_endow_np[:, 0] = 1.0  # Agents only work in their first period
labor_endow = tf.constant(labor_endow_np, dtype=tf.float32)

# Production and household parameters
alpha = tf.constant(0.3)  # Capital share in production
beta_np = 0.7
beta = tf.constant(beta_np, dtype=tf.float32)  # Discount factor (patience)
gamma = tf.constant(1.0, dtype=tf.float32)  # CRRA coefficient

if optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
else:
    raise NotImplementedError



layer = tf.keras.layers.Dense(num_hidden_nodes[0], activation=activations_hidden_nodes[0],
                              kernel_initializer=tf.keras.initializers.RandomNormal(),
                              bias_initializer=tf.keras.initializers.RandomNormal())(X)
for layerIndex in range(1, num_hidden_nodes):
    layer = tf.keras.layers.Dense(num_hidden_nodes[0], activation=activations_hidden_nodes[0])(layer)
savings = tf.keras.layers.Dense(A-1)





def create_training_data():


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







