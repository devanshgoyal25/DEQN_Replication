import os
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np

# tf.config.run_functions_eagerly(True)

print('Tensorflow Version: ', tf.__version__)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('pdf')

from train import *

plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
std_figsize = (4, 4)

def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='TITLE HERE')
    parser.add_argument('--load_episode', type=int, default=0, help='Episode to load weights and starting point from.')
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--plot_interval', type=int, default=20, help='Interval to plot results.')
    parser.add_argument('--save_interval', type=int, default=100, help='Interval to save model.')
    args = parser.parse_args()
    print(args)

    print('##### input arguments #####')
    path_wd = '.'
    num_agents = A = 6 
    num_episodes = 5000
    len_episodes = 10240
    # len_episodes = 200
    epochs_per_episode = 20
    minibatch_size = 512
    # minibatch_size = 100
    num_minibatches = int(len_episodes / minibatch_size)
    lr = 1e-5 
    num_input_nodes = 8 + 4 * A  
    num_hidden_nodes = [100, 50]
    num_output_nodes = (A - 1)
    optimizer_name = 'adam'
    lr = 1e-5

    print('seed: {}'.format(args.seed))
    print('working directory: ' + path_wd)
    print('hidden nodes: [100, 50]')
    print('optimizer: {}'.format(optimizer_name))
    print('batch_size: {}'.format(minibatch_size))
    print('num_episodes: {}'.format(num_episodes))
    print('len_episodes: {}'.format(len_episodes))
    print('epochs_per_episode: {}'.format(epochs_per_episode))
    print('save_interval: {}'.format(args.save_interval))
    print('lr: {}'.format(lr))

    print('###########################')   

    output_path = os.path.join(path_wd, 'output')
    plot_path = os.path.join(output_path, 'plots')
    save_path = os.path.join(output_path, 'models')
    startpoint_path = os.path.join(output_path, 'startpoints')

    if 'output' not in os.listdir(path_wd):
        os.mkdir(output_path)

    if 'plots' not in os.listdir(output_path):
        os.mkdir(plot_path)

    if 'models' not in os.listdir(output_path):
        os.mkdir(save_path)

    if 'startpoints' not in os.listdir(output_path):
        os.mkdir(startpoint_path)

    print('Plots will be saved into ./output/plots/.')

    train(
        A, args.seed, lr, optimizer_name, num_input_nodes,
        num_hidden_nodes, num_output_nodes, minibatch_size,
        num_episodes, len_episodes, epochs_per_episode, path_wd,
        args.save_interval, args.load_episode
    )

if __name__ == '__main__':
    main()
