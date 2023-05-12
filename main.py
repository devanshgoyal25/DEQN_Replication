import os
import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np

# uncomment the folloing when debugging for eager execution
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

    parser = argparse.ArgumentParser(description='DEQN Replication GGR')
    parser.add_argument('--tfp_h', type=float, default=1, help='Probability of the high TFP shock.')
    parser.add_argument('--depr_h', type=float, default=1, help='Probability of the high depreciation shock.')
    parser.add_argument('--load_episode', type=int, default=0, help='Episode to load from.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--plot_interval', type=int, default=20, help='Interval for plotting results.')
    parser.add_argument('--save_interval', type=int, default=20, help='Interval for saving the model.')
    args = parser.parse_args()
    # print(args)

    print('##### input arguments #####')
    path_wd = '.'
    num_agents = A = 6
    num_episodes = 1000
    len_episodes = 10240 # size of simulated dataset per episode
    epochs_per_episode = 20
    minibatch_size = 512
    num_minibatches = int(len_episodes / minibatch_size)
    num_input_nodes = 8 + 4 * A
    num_hidden_nodes = [100, 50]
    num_output_nodes = (A - 1)
    optimizer_name = 'adam'
    lr = 1e-5 # learning rate

    print(f'seed: {args.seed}')
    print(f'working directory: {path_wd}')
    print(f'hidden nodes: {num_hidden_nodes[0]}, {num_hidden_nodes[1]}')
    print(f'optimizer: {optimizer_name}')
    print(f'batch_size: {minibatch_size}')
    print(f'num_episodes: {num_episodes}')
    print(f'len_episodes: {len_episodes}')
    print(f'epochs_per_episode: {epochs_per_episode}')
    print(f'save_interval: {args.save_interval}')
    print(f'learning_rate: {lr}')
    print(f'prob_tfp_high: {args.tfp_h}')
    print(f'prob_depr_high: {args.depr_h}')

    print('###########################')

    # define output directories
    output_path = os.path.join(path_wd, 'output')
    plot_path = os.path.join(output_path, 'plots')
    save_path = os.path.join(output_path, 'models')
    startpoint_path = os.path.join(output_path, 'startpoints')

    # make new directories if they don't exist
    if 'output' not in os.listdir(path_wd):
        os.mkdir(output_path)

    if 'plots' not in os.listdir(output_path):
        os.mkdir(plot_path)

    if 'models' not in os.listdir(output_path):
        os.mkdir(save_path)

    if 'startpoints' not in os.listdir(output_path):
        os.mkdir(startpoint_path)

    print('Plots will be saved to ./output/plots/.')

    # call the DEQN
    train(
        A=A, prob_tfp_high=args.tfp_h, prob_depr_high=args.depr_h, seed=args.seed, lr=lr, optimizer_name=optimizer_name,
        num_input_nodes=num_input_nodes, num_hidden_nodes=num_hidden_nodes, num_output_nodes=num_output_nodes, minibatch_size=minibatch_size,
        num_episodes=num_episodes, len_episodes=len_episodes, epochs_per_episode=epochs_per_episode, path_wd=path_wd,
        save_interval=args.save_interval, load_episode=args.load_episode, plot_interval=args.plot_interval
    )

if __name__ == '__main__':
    main()
