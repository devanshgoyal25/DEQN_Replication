import os
from datetime import datetime
import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)

def train(
    A, seed, lr, optimizer_name, num_input_nodes,
    num_hidden_nodes, num_output_nodes, minibatch_size,
    num_episodes, len_episodes, epochs_per_episode, path_wd,
    save_interval, load_episode
    ):

    #----PARAMS-----
    np.random.seed(seed)
    tf.random.set_seed(seed)

    delta = tf.constant([[0.5], [0.5], [0.9], [0.9]],  dtype=tf.float32)
    eta = tf.constant([[0.95], [1.05], [0.95], [1.05]], dtype=tf.float32)

    p_transition = 0.25
    pi_np = p_transition * np.ones((4, 4))
    pi = tf.constant(pi_np, dtype=tf.float32)

    labor_endow_np = np.zeros((1, A))
    labor_endow_np[:, 0] = 1.0 
    labor_endow = tf.constant(labor_endow_np, dtype=tf.float32)

    alpha = tf.constant(0.3)
    beta_np = 0.7
    beta = tf.constant(beta_np, dtype=tf.float32)
    gamma = tf.constant(1.0, dtype=tf.float32)

    eps = 1e-5
    #---------------

    #-----MODEL-----
    def firm(K, eta, alpha, delta):
        L = tf.ones_like(K)
        r = alpha * eta * K**(alpha - 1) * L**(1 - alpha) + (1 - delta)
        w = (1 - alpha) * eta * K**alpha * L**(-alpha)
        Y = eta * K**alpha * L**(1 - alpha) + (1 - delta) * K
        return r, w, Y

    def shocks(z, eta, delta):
        tfp = tf.gather(eta, tf.cast(z, tf.int32))
        depreciation = tf.gather(delta, tf.cast(z, tf.int32))
        return tfp, depreciation

    def wealth(k, R, l, W):
        fin_wealth = k * R
        lab_wealth = l * W
        tot_income = tf.add(fin_wealth, lab_wealth)
        return fin_wealth, lab_wealth, tot_income

    def get_current_policy(X, net):
        m = tf.shape(X)[0]

        z = X[:, 0] 
        tfp = X[:, 1]
        depr = X[:, 2]
        K = X[:, 3]
        L = X[:, 4]
        r = X[:, 5]
        w = X[:, 6]
        Y = X[:, 7]
        k = X[:, 8 : 8 + A] 
        fw = X[:, 8 + A : 8 + 2 * A] 
        linc = X[:, 8 + 2 * A : 8 + 3 * A] 
        inc = X[:, 8 + 3 * A : 8 + 4 * A] 

        a = net(X)
        a_all = tf.concat([a, tf.zeros([m, 1])], axis = 1)

        c_orig = inc - a_all
        c = tf.maximum(c_orig, tf.ones_like(c_orig) * eps)

        k_prime = tf.concat([tf.zeros([m, 1]), a], axis=1)

        K_prime_orig = tf.reduce_sum(k_prime, axis=1, keepdims=True)
        K_prime = tf.maximum(K_prime_orig, tf.ones_like(K_prime_orig) * eps)

        l_prime = tf.tile(labor_endow, [m, 1])
        L_prime = tf.ones_like(K_prime)

        return c, c_orig, k_prime, K_prime, K_prime_orig, l_prime, L_prime

    def get_next_policy(X, z_next, current_policy, net):
        m = tf.shape(X)[0]
        z = X[:, 0] 

        c, c_orig, k_prime, K_prime, K_prime_orig, l_prime, L_prime = current_policy

        z_prime = z_next * tf.ones_like(z)

        tfp_prime, depr_prime = shocks(z_prime, eta, delta)

        r_prime, w_prime, Y_prime = firm(K_prime, tfp_prime, alpha, depr_prime)
        R_prime = r_prime * tf.ones([1, A])
        W_prime = w_prime * tf.ones([1, A])

        fw_prime, linc_prime, inc_prime = wealth(k_prime, R_prime, l_prime, W_prime)

        x_prime = tf.concat([tf.expand_dims(z_prime, -1),
                               tfp_prime,
                               depr_prime,
                               K_prime,
                               L_prime,
                               r_prime,
                               w_prime,
                               Y_prime,
                               k_prime,
                               fw_prime,
                               linc_prime,
                               inc_prime], axis=1)

        a_prime = net(x_prime)
        a_prime_all = tf.concat([a_prime, tf.zeros([m, 1])], axis=1)

        c_orig_prime = inc_prime - a_prime_all
        c_prime= tf.maximum(c_orig_prime, tf.ones_like(c_orig_prime) * eps)

        return x_prime, R_prime, c_orig_prime, c_prime
    
    def cost(X, net):
        m = tf.shape(X)[0]

        current_policy = get_current_policy(X, net)

        c, c_orig, k_prime, K_prime, K_prime_orig, l_prime, L_prime = current_policy 

        x_prime_1, R_prime_1, c_orig_prime_1, c_prime_1 = get_next_policy(X, 0, current_policy, net)
        x_prime_2, R_prime_2, c_orig_prime_2, c_prime_2 = get_next_policy(X, 1, current_policy, net)
        x_prime_3, R_prime_3, c_orig_prime_3, c_prime_3 = get_next_policy(X, 2, current_policy, net)
        x_prime_4, R_prime_4, c_orig_prime_4, c_prime_4 = get_next_policy(X, 3, current_policy, net)

        pi_trans_to1 = p_transition * tf.ones((m, A-1))
        pi_trans_to2 = p_transition * tf.ones((m, A-1))
        pi_trans_to3 = p_transition * tf.ones((m, A-1))
        pi_trans_to4 = p_transition * tf.ones((m, A-1))

        opt_euler = - 1 + (
            (
                (
                    beta * (
                        pi_trans_to1 * R_prime_1[:, 0:A-1] * c_prime_1[:, 1:A]**(-gamma) 
                        + pi_trans_to2 * R_prime_2[:, 0:A-1] * c_prime_2[:, 1:A]**(-gamma) 
                        + pi_trans_to3 * R_prime_3[:, 0:A-1] * c_prime_3[:, 1:A]**(-gamma) 
                        + pi_trans_to4 * R_prime_4[:, 0:A-1] * c_prime_4[:, 1:A]**(-gamma)
                    )
                ) ** (-1. / gamma)
            ) / c[:, 0:A-1]
        )

        orig_cons = tf.concat([c_orig, c_orig_prime_1, c_orig_prime_2, c_orig_prime_3, c_orig_prime_4], axis=1)
        opt_punish_cons = (1./eps) * tf.maximum(-1 * orig_cons, tf.zeros_like(orig_cons))

        opt_punish_ktot_prime = (1./eps) * tf.maximum(-K_prime_orig, tf.zeros_like(K_prime_orig))

        combined_opt = [opt_euler, opt_punish_cons, opt_punish_ktot_prime]
        opt_predict = tf.concat(combined_opt, axis=1)

        opt_correct = tf.zeros_like(opt_predict)

        mse = tf.keras.losses.MeanSquaredError()

        cost = mse(opt_correct, opt_predict)

        return cost, opt_euler

    #---------------

    #----OPT-----
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise NotImplementedError
    #------------

    #----NET-----
    num_hidden_layer = len(num_hidden_nodes)

    inputs = tf.keras.Input(shape=(A * 4 + 8,))
    hidden1 = tf.keras.layers.Dense(num_hidden_nodes[0], activation='relu')(inputs)
    hidden2 = tf.keras.layers.Dense(num_hidden_nodes[1], activation='relu')(hidden1)
    outputs = tf.keras.layers.Dense(A-1)(hidden2)

    net = tf.keras.Model(inputs=inputs, outputs=outputs)
    net.build(input_shape=(None, A * 4 + 8))
    #------------

    #--TRAIN_STEP--
    parameters = net.trainable_weights

    @tf.function
    def train_step(net, optimizer, X):
        with tf.GradientTape() as tape:
            loss_value = cost(X, net)[0]
        grads = tape.gradient(loss_value, parameters)

        optimizer.apply_gradients(loss_value, parameters)

        return loss_value
    #------------

    #--SIMULATION--
    def simulate_episodes(X_start, episode_length, net):
        time_start = datetime.now()
        print(f"Start simulating {episode_length} periods.")
        dim_state = np.shape(X_start)[1]

        X_episodes = np.zeros([episode_length, dim_state])
        X_episodes[0, :] = X_start
        X_old = tf.convert_to_tensor(X_start)
        print(X_old)

        rand_num = np.random.rand(episode_length, 1)

        for t in range(1, episode_length):
            z = int(X_old[0,0])

            current_policy = get_current_policy(X_old, net)

            if rand_num[t-1] <= pi_np[z,0]:
                X_new = get_next_policy(X_old, 0, current_policy, net)
            elif rand_num[t-1] <= pi_np[z,0] + pi_np[z,1]:
                X_new = get_next_policy(X_old, 1, current_policy, net)
            elif rand_num[t-1] <= pi_np[z,0] + pi_np[z,1] + pi_np[z,2]:
                X_new = get_next_policy(X_old, 2, current_policy, net)
            else:
                X_new = get_next_policy(X_old, 3, current_policy, net)

            X_episodes[t, :] = X_new
            X_old = X_new

        time_end = datetime.now()
        time_diff = time_end - time_start
        print(f"Finished simulation. Time for simulation: {time_diff}.")

        return X_episodes

    def create_minibatches(training_data_X, buffer_size, batch_size, seed):
        train_dataset = tf.data.Dataset.from_tensor_slices(training_data_X)
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=seed)
        return train_dataset
    #--------------

    #---TRAINING---
    train_seed = 0
    buffer_size = len_episodes
    num_batches = buffer_size // minibatch_size

    cost_store, mov_ave_cost_store = [], []

    all_ages = np.arange(1, A+1)
    ages = np.arange(1, A)

    time_start = datetime.now()
    print(f"start time: {time_start}")

    if not(False):
        X_data_train = np.random.rand(1, num_input_nodes)
        X_data_train[:, 0] = (X_data_train[:, 0] > 0.5)
        X_data_train[:, 1:] = X_data_train[:, 1:] + 0.1
        assert np.min(np.sum(X_data_train[:, 1:], axis=1, keepdims=True) > 0) == True, 'Starting point has negative aggregate capital (K)!'
        print('Calculated a valid starting point')

    for episode in range(load_episode, num_episodes):
        X_episodes = simulate_episodes(X_data_train, len_episodes, net)
        X_data_train = X_episodes[-1, :].reshape([1, -1])
        k_dist_mean = np.mean(X_episodes[:, 8 : 8 + A], axis=0)
        k_dist_min = np.min(X_episodes[:, 8 : 8 + A], axis=0)
        k_dist_max = np.max(X_episodes[:, 8 : 8 + A], axis=0)

        ee_error = np.zeros((1, A-1))
        max_ee = np.zeros((1, A-1))

        for epoch in range(epochs_per_episode):
            train_seed += 1
            minibatch_cost = 0

            minibatches = create_minibatches(X_data_train, buffer_size, minibatch_size, seed)

            for minibatch_X in minibatches:
                cost_mini, opt_euler_ = cost(minibatch_X, net)
                minibatch_cost += cost_mini / num_batches

                if epoch == 0:
                    opt_euler_ = np.abs(opt_euler_)
                    ee_error += np.mean(opt_euler_, axis=0) / num_batches
                    mb_max_ee = np.max(opt_euler_, axis=0, keepdims=True)
                    max_ee = np.maximum(max_ee, mb_max_ee)

            if epoch == 0:
                cost_store.append(minibatch_cost)
                mov_ave_cost_store.append(np.mean(cost_store[-100:]))

            for minibatch_X in minibatches:
                train_step(net, optimizer, minibatch_X)
