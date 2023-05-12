import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# uncomment the following when debugging for eager execution
# tf.config.run_functions_eagerly(True)

def train(
    A, prob_tfp_high, prob_depr_high, seed, lr, optimizer_name,
    num_input_nodes, num_hidden_nodes, num_output_nodes, minibatch_size,
    num_episodes, len_episodes, epochs_per_episode, path_wd,
    save_interval, plot_interval, load_episode=0
    ):

    #----PARAMS-----

    # set numpy and tf seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # set shock values
    delta = tf.constant([[0.5], [0.5], [0.9], [0.9]],  dtype=tf.float32)
    eta = tf.constant([[0.95], [1.05], [0.95], [1.05]], dtype=tf.float32)

    # construct transition matrix
    p_tfp_h_depr_h = prob_tfp_high * prob_depr_high
    p_tfp_l_depr_h = (1-prob_tfp_high) * prob_depr_high
    p_tfp_h_depr_l = prob_tfp_high * (1-prob_depr_high)
    p_tfp_l_depr_l = (1-prob_tfp_high) * (1-prob_depr_high)
    pi_row = np.array([p_tfp_h_depr_h, p_tfp_l_depr_h, p_tfp_h_depr_l, p_tfp_l_depr_l])
    pi_np = np.tile(pi_row, (4,1))
    pi = tf.constant(pi_np, dtype=tf.float32)

    # hard-code aggregate labor supply of one as per model
    labor_endow_np = np.zeros((1, A))
    labor_endow_np[:, 0] = 1.0
    labor_endow = tf.constant(labor_endow_np, dtype=tf.float32)

    # model utility parameters
    alpha = tf.constant(0.3)
    beta_np = 0.7
    beta = tf.constant(beta_np, dtype=tf.float32)
    gamma = tf.constant(1.0, dtype=tf.float32)

    # small value for penalty
    eps = 1e-5

    # path for model saves
    model_history_filename = './output/models/model_history.pkl'

    #---------------

    #-----OPT-------

    # configure standard adam optimizer
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    else:
        raise NotImplementedError

    #---------------

    #------NET------

    # number of hidden layers
    num_hidden_layer = len(num_hidden_nodes)

    # default NN weight initializer
    initializer = tf.keras.initializers.GlorotNormal()

    # define NN structure
    inputs = tf.keras.Input(shape=(A * 4 + 8,))
    hidden1 = tf.keras.layers.Dense(num_hidden_nodes[0], activation='relu', kernel_initializer=initializer, bias_initializer=initializer)(inputs)
    hidden2 = tf.keras.layers.Dense(num_hidden_nodes[1], activation='relu', kernel_initializer=initializer, bias_initializer=initializer)(hidden1)
    outputs = tf.keras.layers.Dense(A-1, kernel_initializer=initializer, bias_initializer=initializer, activation='softplus')(hidden2)

    # load-dependent model initialization
    if load_episode==0:
        net = tf.keras.Model(inputs=inputs, outputs=outputs)
        net.build(input_shape=(None, A * 4 + 8))
        model_history = {}
        os.makedirs(os.path.dirname(model_history_filename), exist_ok=True)
    else:
        with open(model_history_filename, 'rb') as handle:
            net = pkl.load(handle)[load_episode]

    #---------------

    #-----MODEL-----

    # model helper function for simulation
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, A * 4 + 8), dtype=tf.float32)])
    def get_next_policies(X):

        # takes capital and turns it into interest rate and wage
        def firm(K, eta, alpha, delta):
            L = tf.ones_like(K) # aggregate labor supply
            r = alpha * eta * K**(alpha - 1) * L**(1 - alpha) + (1 - delta) # rental rate on capital
            w = (1 - alpha) * eta * K**alpha * L**(-alpha) # wage
            Y = eta * K**alpha * L**(1 - alpha) + (1 - delta) * K # firm output
            return r, w, Y

        # given shock index, return values of tfp and depr
        def shocks(z, eta, delta):
            tfp = tf.gather(eta, tf.cast(z, tf.int32))
            depreciation = tf.gather(delta, tf.cast(z, tf.int32))
            return tfp, depreciation

        # given capital and interest rate, return income by source
        def wealth(k, R, l, W):
            fin_wealth = k * R
            lab_wealth = l * W
            tot_income = tf.add(fin_wealth, lab_wealth)
            return fin_wealth, lab_wealth, tot_income

        # record size and shape of input data
        m = tf.shape(X)[0]
        X = tf.cast(X, dtype=tf.float32)

        # transliterate the input
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

        # use the NN to get today's savings
        a = net(X)
        a_all = tf.concat([a, tf.zeros([m, 1])], axis = 1)

        # calculate consumption from savings
        c_orig = inc - a_all
        c = tf.maximum(c_orig, tf.ones_like(c_orig) * eps)

        # find individual and aggregate capital
        k_prime = tf.concat([tf.zeros([m, 1]), a], axis=1)
        K_prime_orig = tf.reduce_sum(k_prime, axis=1, keepdims=True)
        K_prime = tf.maximum(K_prime_orig, tf.ones_like(K_prime_orig) * eps)

        # labor endowmnets
        l_prime = tf.tile(labor_endow, [m, 1])
        L_prime = tf.ones_like(K_prime)

        #--PREDICT SAVINGS FOR SHOCK 1--

        # shock index definition
        z_prime_1 = 0 * tf.ones_like(z)

        # find shock values
        tfp_prime_1, depr_prime_1 = shocks(z_prime_1, eta, delta)

        # find predicted interest rates and wages
        r_prime_1, w_prime_1, Y_prime_1 = firm(K_prime, tfp_prime_1, alpha, depr_prime_1)
        R_prime_1 = r_prime_1 * tf.ones([1, A])
        W_prime_1 = w_prime_1 * tf.ones([1, A])

        # find predicted income
        fw_prime_1, linc_prime_1, inc_prime_1 = wealth(k_prime, R_prime_1, l_prime, W_prime_1)

        # input for the NN to predict shock savings
        x_prime_1 = tf.concat([tf.expand_dims(z_prime_1, -1),
                        tfp_prime_1,
                        depr_prime_1,
                        K_prime,
                        L_prime,
                        r_prime_1,
                        w_prime_1,
                        Y_prime_1,
                        k_prime,
                        fw_prime_1,
                        linc_prime_1,
                        inc_prime_1], axis=1)

        #--PREDICT SAVINGS FOR SHOCK 2-- (see comments above)

        z_prime_2 = 1 * tf.ones_like(z)

        tfp_prime_2, depr_prime_2 = shocks(z_prime_2, eta, delta)

        r_prime_2, w_prime_2, Y_prime_2 = firm(K_prime, tfp_prime_2, alpha, depr_prime_2)
        R_prime_2 = r_prime_2 * tf.ones([1, A])
        W_prime_2 = w_prime_2 * tf.ones([1, A])

        fw_prime_2, linc_prime_2, inc_prime_2 = wealth(k_prime, R_prime_2, l_prime, W_prime_2)

        x_prime_2 = tf.concat([tf.expand_dims(z_prime_2, -1),
                        tfp_prime_2,
                        depr_prime_2,
                        K_prime,
                        L_prime,
                        r_prime_2,
                        w_prime_2,
                        Y_prime_2,
                        k_prime,
                        fw_prime_2,
                        linc_prime_2,
                        inc_prime_2], axis=1)

        #--PREDICT SAVINGS FOR SHOCK 3-- (see comments above)

        z_prime_3 = 2 * tf.ones_like(z)

        tfp_prime_3, depr_prime_3 = shocks(z_prime_3, eta, delta)

        r_prime_3, w_prime_3, Y_prime_3 = firm(K_prime, tfp_prime_3, alpha, depr_prime_3)
        R_prime_3 = r_prime_3 * tf.ones([1, A])
        W_prime_3 = w_prime_3 * tf.ones([1, A])

        fw_prime_3, linc_prime_3, inc_prime_3 = wealth(k_prime, R_prime_3, l_prime, W_prime_3)

        x_prime_3 = tf.concat([tf.expand_dims(z_prime_3, -1),
                        tfp_prime_3,
                        depr_prime_3,
                        K_prime,
                        L_prime,
                        r_prime_3,
                        w_prime_3,
                        Y_prime_3,
                        k_prime,
                        fw_prime_3,
                        linc_prime_3,
                        inc_prime_3], axis=1)

        #--PREDICT SAVINGS FOR SHOCK 4-- (see comments above)

        z_prime_4 = 3 * tf.ones_like(z)

        tfp_prime_4, depr_prime_4 = shocks(z_prime_4, eta, delta)

        r_prime_4, w_prime_4, Y_prime_4 = firm(K_prime, tfp_prime_4, alpha, depr_prime_4)
        R_prime_4 = r_prime_4 * tf.ones([1, A])
        W_prime_4 = w_prime_4 * tf.ones([1, A])

        fw_prime_4, linc_prime_4, inc_prime_4 = wealth(k_prime, R_prime_4, l_prime, W_prime_4)

        x_prime_4 = tf.concat([tf.expand_dims(z_prime_4, -1),
                        tfp_prime_4,
                        depr_prime_4,
                        K_prime,
                        L_prime,
                        r_prime_4,
                        w_prime_4,
                        Y_prime_4,
                        k_prime,
                        fw_prime_4,
                        linc_prime_4,
                        inc_prime_4], axis=1)

        return x_prime_1, x_prime_2, x_prime_3, x_prime_4

    # model helper function for training
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, A * 4 + 8), dtype=tf.float32)])
    def cost(X):

        # takes capital and turns it into interest rate and wage
        def firm(K, eta, alpha, delta):
            L = tf.ones_like(K) # aggregate labor supply
            r = alpha * eta * K**(alpha - 1) * L**(1 - alpha) + (1 - delta) # rental rate on capital
            w = (1 - alpha) * eta * K**alpha * L**(-alpha) # wage
            Y = eta * K**alpha * L**(1 - alpha) + (1 - delta) * K # firm output
            return r, w, Y

        # given shock index, return values of tfp and depr
        def shocks(z, eta, delta):
            tfp = tf.gather(eta, tf.cast(z, tf.int32))
            depreciation = tf.gather(delta, tf.cast(z, tf.int32))
            return tfp, depreciation

        # given capital and interest rate, return income by source
        def wealth(k, R, l, W):
            fin_wealth = k * R
            lab_wealth = l * W
            tot_income = tf.add(fin_wealth, lab_wealth)
            return fin_wealth, lab_wealth, tot_income

        # record size and shape of input data
        m = tf.shape(X)[0]
        X = tf.cast(X, dtype=tf.float32)

        # transliterate the input
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

        # use the NN to get today's savings
        a = net(X)
        a_all = tf.concat([a, tf.zeros([m, 1])], axis = 1)

        # calculate consumption from savings
        c_orig = inc - a_all
        c = tf.maximum(c_orig, tf.ones_like(c_orig) * eps)

        # find individual and aggregate capital
        k_prime = tf.concat([tf.zeros([m, 1]), a], axis=1)
        K_prime_orig = tf.reduce_sum(k_prime, axis=1, keepdims=True)
        K_prime = tf.maximum(K_prime_orig, tf.ones_like(K_prime_orig) * eps)

        # labor endowmnets
        l_prime = tf.tile(labor_endow, [m, 1])
        L_prime = tf.ones_like(K_prime)

        #--PREDICT SAVINGS FOR SHOCK 1--

        # shock index definition
        z_prime_1 = 0 * tf.ones_like(z)

        # find shock values
        tfp_prime_1, depr_prime_1 = shocks(z_prime_1, eta, delta)

        # find predicted interest rates and wages
        r_prime_1, w_prime_1, Y_prime_1 = firm(K_prime, tfp_prime_1, alpha, depr_prime_1)
        R_prime_1 = r_prime_1 * tf.ones([1, A])
        W_prime_1 = w_prime_1 * tf.ones([1, A])

        # find predicted income
        fw_prime_1, linc_prime_1, inc_prime_1 = wealth(k_prime, R_prime_1, l_prime, W_prime_1)

        # input for the NN to predict shock savings
        x_prime_1 = tf.concat([tf.expand_dims(z_prime_1, -1),
                        tfp_prime_1,
                        depr_prime_1,
                        K_prime,
                        L_prime,
                        r_prime_1,
                        w_prime_1,
                        Y_prime_1,
                        k_prime,
                        fw_prime_1,
                        linc_prime_1,
                        inc_prime_1], axis=1)


        # predict the shock savings
        a_prime_1 = net(x_prime_1)
        a_prime_all_1 = tf.concat([a_prime_1, tf.zeros([m, 1])], axis=1)

        # infer consumption
        c_orig_prime_1 = inc_prime_1 - a_prime_all_1
        c_prime_1 = tf.maximum(c_orig_prime_1, tf.ones_like(c_orig_prime_1) * eps)

        #--PREDICT SAVINGS FOR SHOCK 2-- (see comments above)

        z_prime_2 = 1 * tf.ones_like(z)

        tfp_prime_2, depr_prime_2 = shocks(z_prime_2, eta, delta)

        r_prime_2, w_prime_2, Y_prime_2 = firm(K_prime, tfp_prime_2, alpha, depr_prime_2)
        R_prime_2 = r_prime_2 * tf.ones([1, A])
        W_prime_2 = w_prime_2 * tf.ones([1, A])

        fw_prime_2, linc_prime_2, inc_prime_2 = wealth(k_prime, R_prime_2, l_prime, W_prime_2)

        x_prime_2 = tf.concat([tf.expand_dims(z_prime_2, -1),
                        tfp_prime_2,
                        depr_prime_2,
                        K_prime,
                        L_prime,
                        r_prime_2,
                        w_prime_2,
                        Y_prime_2,
                        k_prime,
                        fw_prime_2,
                        linc_prime_2,
                        inc_prime_2], axis=1)

        a_prime_2 = net(x_prime_2)
        a_prime_all_2 = tf.concat([a_prime_2, tf.zeros([m, 1])], axis=1)

        c_orig_prime_2 = inc_prime_2 - a_prime_all_2
        c_prime_2 = tf.maximum(c_orig_prime_2, tf.ones_like(c_orig_prime_2) * eps)

        #--PREDICT SAVINGS FOR SHOCK 3-- (see comments above)

        z_prime_3 = 2 * tf.ones_like(z)

        tfp_prime_3, depr_prime_3 = shocks(z_prime_3, eta, delta)

        r_prime_3, w_prime_3, Y_prime_3 = firm(K_prime, tfp_prime_3, alpha, depr_prime_3)
        R_prime_3 = r_prime_3 * tf.ones([1, A])
        W_prime_3 = w_prime_3 * tf.ones([1, A])

        fw_prime_3, linc_prime_3, inc_prime_3 = wealth(k_prime, R_prime_3, l_prime, W_prime_3)

        x_prime_3 = tf.concat([tf.expand_dims(z_prime_3, -1),
                        tfp_prime_3,
                        depr_prime_3,
                        K_prime,
                        L_prime,
                        r_prime_3,
                        w_prime_3,
                        Y_prime_3,
                        k_prime,
                        fw_prime_3,
                        linc_prime_3,
                        inc_prime_3], axis=1)

        a_prime_3 = net(x_prime_3)
        a_prime_all_3 = tf.concat([a_prime_3, tf.zeros([m, 1])], axis=1)

        c_orig_prime_3 = inc_prime_3 - a_prime_all_3
        c_prime_3 = tf.maximum(c_orig_prime_3, tf.ones_like(c_orig_prime_3) * eps)

        #--PREDICT SAVINGS FOR SHOCK 4-- (see comments above)

        z_prime_4 = 3 * tf.ones_like(z)

        tfp_prime_4, depr_prime_4 = shocks(z_prime_4, eta, delta)

        r_prime_4, w_prime_4, Y_prime_4 = firm(K_prime, tfp_prime_4, alpha, depr_prime_4)
        R_prime_4 = r_prime_4 * tf.ones([1, A])
        W_prime_4 = w_prime_4 * tf.ones([1, A])

        fw_prime_4, linc_prime_4, inc_prime_4 = wealth(k_prime, R_prime_4, l_prime, W_prime_4)

        x_prime_4 = tf.concat([tf.expand_dims(z_prime_4, -1),
                        tfp_prime_4,
                        depr_prime_4,
                        K_prime,
                        L_prime,
                        r_prime_4,
                        w_prime_4,
                        Y_prime_4,
                        k_prime,
                        fw_prime_4,
                        linc_prime_4,
                        inc_prime_4], axis=1)

        a_prime_4 = net(x_prime_4)
        a_prime_all_4 = tf.concat([a_prime_4, tf.zeros([m, 1])], axis=1)

        c_orig_prime_4 = inc_prime_4 - a_prime_all_4
        c_prime_4 = tf.maximum(c_orig_prime_4, tf.ones_like(c_orig_prime_4) * eps)

        #--CALCULATE COST--

        # code transition probabilities
        pi_trans_to1 = p_tfp_h_depr_h * tf.ones((m, A-1))
        pi_trans_to2 = p_tfp_l_depr_h * tf.ones((m, A-1))
        pi_trans_to3 = p_tfp_h_depr_l * tf.ones((m, A-1))
        pi_trans_to4 = p_tfp_l_depr_l * tf.ones((m, A-1))

        # get error for the relative euler equation
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

        # punishment for negative consumption values
        orig_cons = tf.concat([c_orig, c_orig_prime_1, c_orig_prime_2, c_orig_prime_3, c_orig_prime_4], axis=1)
        opt_punish_cons = (1./eps) * tf.maximum(-1 * orig_cons, tf.zeros_like(orig_cons))

        # punishment for negative capital values
        opt_punish_ktot_prime = (1./eps) * tf.maximum(-K_prime_orig, tf.zeros_like(K_prime_orig))

        # overall cost
        combined_opt = [opt_euler, opt_punish_cons, opt_punish_ktot_prime]
        opt_predict = tf.concat(combined_opt, axis=1)

        # the goal cost is zero
        opt_correct = tf.zeros_like(opt_predict)

        # compare goal to empirical
        mse = tf.keras.losses.MeanSquaredError()

        # loss function overall
        cost = mse(opt_correct, opt_predict)

        return cost, opt_euler

    #----------------

    #---TRAIN_STEP---

    # get the model weights
    parameters = net.trainable_weights

    # take a training step
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, A * 4 + 8), dtype=tf.float32)])
    def train_step(X):

        # calculate loss with autodiff and clip gradients
        with tf.GradientTape() as tape:
            loss_value = cost(X)[0]
        grads = tape.gradient(loss_value, parameters)
        grads = [(tf.clip_by_value(grad, -1.,1.)) for grad in grads]

        # take the actual step
        optimizer.apply_gradients(zip(grads, parameters))

        return loss_value

    #-------------

    #--SIMULATION--

    # given a starting point, simulate episode_length episodes
    def simulate_episodes(X_start, episode_length, net):
        time_start = datetime.now()
        print(f"Start simulating {episode_length} periods.")
        dim_state = np.shape(X_start)[1]

        X_episodes = np.zeros([episode_length, dim_state])
        X_episodes[0, :] = X_start
        X_old = tf.convert_to_tensor(X_start, dtype=tf.float32)

        rand_num = np.random.rand(episode_length, 1)

        for t in range(1, episode_length):
            z = int(X_old[0,0])

            X_predicts = get_next_policies(X_old)

            if rand_num[t-1] <= pi_np[z,0]:
                X_new = X_predicts[0]
            elif rand_num[t-1] <= pi_np[z,0] + pi_np[z,1]:
                X_new = X_predicts[1]
            elif rand_num[t-1] <= pi_np[z,0] + pi_np[z,1] + pi_np[z,2]:
                X_new = X_predicts[2]
            else:
                X_new = X_predicts[3]

            X_episodes[t, :] = X_new
            X_old = X_new

        time_end = datetime.now()
        time_diff = time_end - time_start
        print(f"Finished simulation. Time for simulation: {time_diff}.")

        return X_episodes

    # given a dataset, create random minibatches
    def create_minibatches(training_data_X, buffer_size, batch_size, seed):
        train_dataset = tf.data.Dataset.from_tensor_slices(training_data_X)
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size)
        return train_dataset

    #--------------

    #-----ANALYTIC SOLUTION -----
    
    # get the analytic solution to the model
    @tf.function(reduce_retracing=True)
    def get_analytic(X):
        inc = X[:, 8 + 3 * A : 8 + 4 * A]
        beta_vec = beta_np * (1 - beta_np ** (A - 1 - np.arange(A - 1))) / (1 - beta_np ** (A - np.arange(A - 1)))
        beta_vec = tf.constant(np.expand_dims(beta_vec, 0), dtype=tf.float64)
        a_analytic = inc[:, : -1] * beta_vec
        return a_analytic

    #--------------

    #---TRAINING---

    # set the running seed for batching, along with params
    train_seed = 0
    buffer_size = len_episodes
    num_batches = buffer_size // minibatch_size

    # plotting lists
    cost_store, mov_ave_cost_store = [], []

    # ploting helper vars
    all_ages = np.arange(1, A+1)
    ages = np.arange(1, A)

    time_start = datetime.now()
    print(f"start time: {time_start}")

    # generate simulation start point
    X_data_train = np.random.rand(1, num_input_nodes)
    X_data_train[:, 0] = (X_data_train[:, 0] > 0.5)
    X_data_train[:, 1:] = X_data_train[:, 1:] + 0.1
    assert np.min(np.sum(X_data_train[:, 1:], axis=1, keepdims=True) > 0) == True, 'Starting point has negative aggregate capital (K)!'
    print('Calculated a valid starting point')

    # plotting params
    plt.rcParams.update({'font.size': 12})
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    std_figsize = (4, 4)

    # training loop begins
    for episode in range(load_episode, num_episodes):
        # simulate dataset
        print(f"Start training for {epochs_per_episode} epochs.")
        X_episodes = simulate_episodes(X_data_train, len_episodes, net)
        X_data_train = X_episodes[-1, :].reshape([1, -1])
        k_dist_mean = np.mean(X_episodes[:, 8 : 8 + A], axis=0)
        k_dist_min = np.min(X_episodes[:, 8 : 8 + A], axis=0)
        k_dist_max = np.max(X_episodes[:, 8 : 8 + A], axis=0)

        ee_error = np.zeros((1, A-1))
        max_ee = np.zeros((1, A-1))

        # minibatch gradient descent loop
        for epoch in range(epochs_per_episode):
            train_seed += 1
            minibatch_cost = 0

            minibatches = create_minibatches(X_episodes, buffer_size, minibatch_size, seed)

            for step, minibatch_X in enumerate(minibatches):
                cost_mini, opt_euler_ = cost(tf.cast(minibatch_X, dtype=tf.float32))
                minibatch_cost += cost_mini / num_batches

                if epoch == 0:
                    opt_euler_ = np.abs(opt_euler_)
                    ee_error += np.mean(opt_euler_, axis=0) / num_batches
                    mb_max_ee = np.max(opt_euler_, axis=0, keepdims=True)
                    max_ee = np.maximum(max_ee, mb_max_ee)

            if epoch == 0:
                cost_store.append(minibatch_cost)
                mov_ave_cost_store.append(np.mean(cost_store[-100:]))

            for step, minibatch_X in enumerate(minibatches):
                train_step(tf.cast(minibatch_X, dtype=tf.float32))

        print(f"Episode {episode} done.")
        print(f"Episode: {episode} \t Cost: {np.log10(cost_store[-1])}")

        # save the model into a .pkl every 20 episodes (default)
        if episode % save_interval == 0:
            if load_episode ==0:
                model_history[episode] = net
            else:
                with open(model_history_filename, 'rb') as handle:
                    model_history = pkl.load(handle)
                model_history[episode] = net

            with open(model_history_filename, 'wb') as handle:
                pkl.dump(model_history, handle)

        if episode % plot_interval == 0:
            # plot the loss function
            plt.figure(figsize=std_figsize)
            plt.clf()
            ax = plt.subplot(1, 1, 1)
            ax.plot(np.log10(cost_store), 'k-', label='cost')
            ax.plot(np.log10(mov_ave_cost_store), 'r--', label='moving average')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Cost [log10]')
            ax.legend(loc='upper right')
            plt.savefig('./output/plots/loss_ep_%d.pdf' % episode, bbox_inches='tight')
            plt.close()

            # plot the relative Euler equation losses
            plt.figure(figsize=std_figsize)
            plt.clf()
            ax = plt.subplot(1, 1, 1)
            ax.plot(ages, np.log10(ee_error).ravel(), 'k-', label='mean')
            ax.plot(ages, np.log10(max_ee).ravel(), 'k--', label='max')
            ax.set_xlabel('Age')
            ax.set_ylabel('Rel EE [log10]')
            ax.legend()
            plt.savefig('./output/plots/relee_ep_%d.pdf' % episode, bbox_inches='tight')
            plt.close()

            # plot capital distribution by age
            plt.figure(figsize=std_figsize)
            plt.clf()
            ax = plt.subplot(1, 1, 1)
            ax.plot(all_ages, k_dist_mean, 'k-', label='mean')
            ax.plot(all_ages, k_dist_min, 'k-.', label='min')
            ax.plot(all_ages, k_dist_max, 'k--', label='max')
            ax.set_xlabel('Age')
            ax.set_ylabel('Capital (k)')
            ax.legend()
            ax.set_xticks(all_ages)
            plt.savefig('./output/plots/distk_ep_%d.pdf' % episode, bbox_inches='tight')
            plt.close()

            # sample states and compare performance to the analytic solution
            pick = np.random.randint(len_episodes, size=50)
            random_states = X_episodes[pick, :]

            # sort states by shock
            random_states_1 = random_states[random_states[:, 0] == 0]
            random_states_2 = random_states[random_states[:, 0] == 1]
            random_states_3 = random_states[random_states[:, 0] == 2]
            random_states_4 = random_states[random_states[:, 0] == 3]

            # compute the capital distribution for each state
            random_k_1 = random_states_1[:, 8: 8 + A]
            random_k_2 = random_states_2[:, 8: 8 + A]
            random_k_3 = random_states_3[:, 8: 8 + A]
            random_k_4 = random_states_4[:, 8: 8 + A]

            # approximate a solution with the neural network
            nn_pred_1 = net(random_states_1)
            nn_pred_2 = net(random_states_2)
            nn_pred_3 = net(random_states_3)
            nn_pred_4 = net(random_states_4)

            # calculate the solution analytically
            true_pol_1 = get_analytic(random_states_1)
            true_pol_2 = get_analytic(random_states_2)
            true_pol_3 = get_analytic(random_states_3)
            true_pol_4 = get_analytic(random_states_4)

            for i in range(A - 1):
                plt.figure(figsize=std_figsize)
                ax = plt.subplot(1, 1, 1)
                
                # plot the analytical solution with a circle
                ax.plot(random_k_1[:, i], true_pol_1[:, i], 'ro', mfc='none', alpha=0.5, markersize=6, label='analytic')
                ax.plot(random_k_2[:, i], true_pol_2[:, i], 'bo', mfc='none', alpha=0.5, markersize=6)
                ax.plot(random_k_3[:, i], true_pol_3[:, i], 'go', mfc='none', alpha=0.5, markersize=6)
                ax.plot(random_k_4[:, i], true_pol_4[:, i], 'yo', mfc='none', alpha=0.5, markersize=6)
                
                # plot the prediction by the neural network with a dot
                ax.plot(random_k_1[:, i], nn_pred_1[:, i], 'r*', markersize=2, label='DEQN')
                ax.plot(random_k_2[:, i], nn_pred_2[:, i], 'b*', markersize=2)
                ax.plot(random_k_3[:, i], nn_pred_3[:, i], 'g*', markersize=2)
                ax.plot(random_k_4[:, i], nn_pred_4[:, i], 'y*', markersize=2)
                ax.set_title('Agent {}'.format(i + 1))
                ax.set_xlabel(r'$k_t$')
                ax.set_ylabel(r'$a_t$')
                ax.legend()
                plt.savefig('./output/plots/policy_agent_%d_ep_%d.pdf' % (i + 1, episode), bbox_inches='tight')
                plt.close()
