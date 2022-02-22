import sys
sys.path.append('../') # Add DMGP root

import os
import os.path as osp
import copy
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
rng = default_rng() # Add seed if needed

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import gpflow
from gpflow import kullback_leiblers
from gpflow.models import GPModel
from gpflow.models.util import inducingpoint_wrapper
from gpflow.inducing_variables import InducingPoints
from gpflow.base import Parameter
from gpflow.kullback_leiblers import gauss_kl
from gpflow.kernels import SquaredExponential
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingPoints
from gpflow.likelihoods import Gaussian
from gpflow.utilities import (
    print_summary, 
    set_trainable, 
    to_default_float, 
    triangular, 
    positive
)
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
    ExecuteCallback
)

import dmgp
from dmgp.model import DMGP
from dmgp.core import vec, vec_inv, KP, ind_kmeans

gpflow.config.set_default_jitter(1e-6)

# CONSTANTS
TRAIN_PARAMS_PATH = './params/train_dmgp_params.yaml'

# Returns a mini-batch given data and indices
def get_batch(data, ids, idx_split, mask=None):
    X, G, Y = data
    G_idx_split, X_idx_split = idx_split

    X_batch = X[X_idx_split,:]
    G_batch = G[G_idx_split,:]
    
    ix = np.ix_(G_idx_split, X_idx_split)
    Y_batch = Y[ix]

    G_batch_ids = ids[0][G_idx_split]
    X_batch_ids = ids[1][X_idx_split]

    if mask is not None:
        mask_batch = mask[ix]
        return (X_batch, G_batch, Y_batch), (G_batch_ids, X_batch_ids), mask_batch
    else:
        return (X_batch, G_batch, Y_batch), (G_batch_ids, X_batch_ids)

# Splits the indices into chunks and generates pairs
def get_idx_splits(G_idx, X_idx, b, a):
    G_idx_splits = np.split(G_idx, np.arange(b, len(G_idx), b))
    X_idx_splits = np.split(X_idx, np.arange(a, len(X_idx), a))
    idx_splits = [(G_idx_split, X_idx_split) 
                  for G_idx_split in G_idx_splits 
                  for X_idx_split in X_idx_splits]

    return idx_splits

# Defines a single optimization step i.e. gradient update
def opt_step(model, batch, batch_ids, optimizer, mask=None):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch, batch_ids, mask=mask)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss

# Trains the given model
# Note: For TensorBoard use, pass gpflow.monitor.Monitor() argument
def train_model(model, data, ids, batch_shape, optimizer, epochs=100, log_freq=1, ckpt_freq=50, conv=1e-4, window=10, patience=10,
                min_n_epochs=50, monitor=None, manager=None, epoch_var=None, step_var=None, start_time=None, mask=None):
    tf_opt_step = tf.function(opt_step)
    
    elbos = []
    avg_decrease = 0. # Initial average decrease
    wait = 0
    conv_cnt = 0
    decay = 1. - 1. / window
    prev_avg_epoch_loss = 0
    assert(min_n_epochs <= epochs and min_n_epochs >= window)

    p_train, n_train = data[-1].shape
    b, a = batch_shape
    G_train_ids, X_train_ids = ids
    G_train_idx = np.array(range(p_train))
    X_train_idx = np.array(range(n_train))
    idx_splits = get_idx_splits(G_train_idx, X_train_idx, b, a)

    for epoch in range(epochs):
        epoch_loss = tf.convert_to_tensor(tf.Variable(0, dtype=default_float()))
        epoch_id = epoch + 1
        
        for step, idx_split in enumerate(tqdm(idx_splits, desc='Batches')):
            if mask is not None:
                batch, batch_ids, mask_batch = get_batch(data, ids, idx_split, mask=mask)
                epoch_loss = epoch_loss + tf_opt_step(model, batch, batch_ids, optimizer, mask=mask_batch)
            else:
                batch, batch_ids = get_batch(data, ids, idx_split)
                epoch_loss = epoch_loss + tf_opt_step(model, batch, batch_ids, optimizer)
            
            if step_var is not None:
                step_var.assign(epoch * len(idx_splits) + step + 1)
            
        if epoch_var is not None:
            epoch_var.assign(epoch + 1)

        epoch_loss = epoch_loss / len(idx_splits) # Take average over steps
        elbos.append(-epoch_loss)

        # Record elapsed time at epoch
        time_at_epoch = (datetime.now() - start_time).seconds

        '''
        if len(elbos) > 0:
            epoch_loss = epoch_loss / len(idx_splits) # Compute average loss within epoch
            prev_loss = -elbos[-1] # Note: Current epoch loss not appended yet
            epoch_decrease = prev_loss - epoch_loss
            avg_decrease = epoch_decrease + decay * (avg_decrease - epoch_decrease)
        '''

        if epoch_id % log_freq == 0:
            #elbo = -epoch_loss
            #kl = model.total_prior_kl()
            #exp = elbo + kl

            if monitor is not None:
                monitor(epoch, epoch_id=epoch_id, elbo=-epoch_loss, time=time_at_epoch, prior_kl=model.total_prior_kl())
            else:
                tf.print(f'[Epoch {epoch_id}] ELBO: {-epoch_loss:.4f}\tPrior KL: {model.total_prior_kl():.4f}')
                
        if epoch_id % ckpt_freq == 0:
            if manager is not None:
                ckpt_path = manager.save()
                tf.print(f'Model checkpoint saved at: {ckpt_path}.')

        if epoch > 0 and epoch % window == 0 and wait == 0:
            avg_epoch_loss = -np.mean(elbos[-window:])
            #abs_diff = np.abs(avg_epoch_loss - prev_avg_epoch_loss)
            abs_percent_change = np.abs((avg_epoch_loss - prev_avg_epoch_loss) / prev_avg_epoch_loss)
            #print(abs_diff, abs_percent_change) # TESTING

            if abs_percent_change < conv:
                conv_cnt += 1

                if conv_cnt > patience:
                    tf.print(f'[LossNotDecreasing] Convergence reached, patience exhausted. Stopping training.')
                    break
                
                else:
                    tf.print(f'[LossNotDecreasing] Tracking convergence. Patience left={patience-conv_cnt}.')
                    wait = 10

            prev_avg_epoch_loss = avg_epoch_loss

        elif wait > 0:
            wait -= 1

        # Shuffle data indices
        np.random.shuffle(G_train_idx)
        np.random.shuffle(X_train_idx)
        idx_splits = get_idx_splits(G_train_idx, X_train_idx, b, a)
        
    return elbos

# Trains DMGP on a given dataset with specified parameters
def train_dmgp(train_params, main_log_dir, group_idxs=None, input_idxs=None, show_summary=False, exp_id=None, miss_flag=False):
    train_path = train_params['train_path']
    test_path = train_params['test_path']
    mask_path = train_params['mask_path']
    b, a = train_params['batch_shape']
    mX = train_params['mX']
    mG = train_params['mG']
    lr = float(train_params['lr'])
    epochs = train_params['epochs']
    fix_ind = bool(train_params['fix_ind'])
    acc_metrics = train_params['acc_metrics'] # Accuracy metrics ('mse', 'mae', 'rmse', 'nlpd')
    init_method = train_params['init_method'] # Initialization ('subset', 'kmeans')
    dataset = osp.basename(osp.dirname(train_path))
    train_npz = np.load(train_path)
    test_npz = np.load(test_path)

    # Logging directory
    exp_str = f'{dataset}' + f'_R{exp_id}' if exp_id is not None else f'{dataset}'
    log_dir = osp.join(main_log_dir, exp_str)

    if not osp.isdir(log_dir):
        os.makedirs(log_dir)

    if miss_flag:
        print(f'Training DMGP for missing value prediction...')
    else:
        print(f'Training DMGP for held-out data prediction...')

    # Unpack data
    X_train = train_npz['X']
    X_test = test_npz['X']
    G_train = train_npz['G']
    G_test = test_npz['G']
    Y_train = train_npz['Y']
    Y_test = test_npz['Y']

    X_train_ids = train_npz['X_ids']
    X_test_ids = test_npz['X_ids']
    G_train_ids = train_npz['G_ids']
    G_test_ids = test_npz['G_ids']

    # If performing missing value prediction
    if miss_flag:
        mask_npz = np.load(mask_path)
        data_mask = mask_npz['mask']
        assert(data_mask.shape == Y_train.shape)
    else:
        data_mask = None

    # Gather data
    train_data = (X_train, G_train, Y_train)
    test_data = (X_test, G_test, Y_test)

    # Gather task & sample IDs
    train_ids = (G_train_ids, X_train_ids)
    test_ids = (G_test_ids, X_test_ids)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p_train = G_train.shape[0]
    p_test = G_test.shape[0]

    # Initialize inducing points
    if init_method == 'subset':
        print(f'Initializing inducing points with subset of training inputs...')
        inds = (InducingPoints(X_train[rng.choice(n_train, size=mX, replace=False),:], name='ZX_fixed'),
                InducingPoints(X_train[rng.choice(n_train, size=mX, replace=False),:], name='ZX_random'),
                InducingPoints(G_train[rng.choice(p_train, size=mG, replace=False),:], name='ZG_fixed'),
                InducingPoints(G_train[rng.choice(p_train, size=mG, replace=False),:], name='ZG_random'))
    elif init_method == 'kmeans':
        print(f'Initializing inducing points with K-means clusters...')
        X_inds = ind_kmeans(mX, X_train)
        G_inds = ind_kmeans(mG, G_train)

        inds = (InducingPoints(X_inds, name='ZX_fixed'),
                InducingPoints(X_inds, name='ZX_random'),
                InducingPoints(G_inds, name='ZG_fixed'),
                InducingPoints(G_inds, name='ZG_random'))

    # Instantiate model
    model = DMGP(Y_train.shape, train_ids, kernel=SquaredExponential(), ll=Gaussian(), ind_points=inds)

    if fix_ind:
        gpflow.set_trainable(model.ZX_fixed, False)
        gpflow.set_trainable(model.ZX_random, False)
        gpflow.set_trainable(model.ZG_fixed, False)
        gpflow.set_trainable(model.ZG_random, False)
        
    if show_summary:
        print_summary(model)

    # Specify monitor object
    keywords = ['KX_fixed', 'KX_random', 'KG_fixed', 'KG_random', 'likelihood']
    model_task = ModelToTensorBoard(log_dir, model, keywords_to_monitor=keywords)
    elbo_task = ScalarToTensorBoard(log_dir, dmgp.core.avg_elbo_cb, 'ELBO')
    time_task = ScalarToTensorBoard(log_dir, dmgp.core.time_cb, 'elapsed_time_at_epoch')
    print_task = ExecuteCallback(callback=dmgp.core.print_avg_elbo)

    monitor = Monitor(MonitorTaskGroup([model_task, elbo_task, time_task, print_task]))

    # Specify checkpoint details
    step_var = tf.Variable(1, dtype=tf.int32, trainable=False)
    epoch_var = tf.Variable(1, dtype=tf.int32, trainable=False)
    ckpt = tf.train.Checkpoint(model=model, step=step_var, epoch=epoch_var)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)

    # Specify optimizer
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    start_time = datetime.now()

    # Train model with monitoring and checkpointing
    elbos = train_model(model, (X_train,G_train,Y_train), (G_train_ids,X_train_ids), (b,a),
                        optimizer, epochs=epochs, monitor=monitor, ckpt_freq=50, 
                        manager=manager, epoch_var=epoch_var, step_var=step_var,
                        start_time=start_time, mask=data_mask)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    hr, rem = divmod(elapsed_time.seconds, 3600)
    mins, secs = divmod(rem, 60)
    n_epochs = len(elbos)
    print(f'\nTime until convergence: {hr}hr, {mins}min, {secs}s ({n_epochs} epochs).\n')

    # Record training time
    # Note: runtime.txt records final code runtime
    with open(osp.join(log_dir, 'runtime.txt'), 'w') as fh:
        fh.write(f'Elapsed Time: {str(elapsed_time)} ({n_epochs} epochs)')

    # Generate ELBO plot and save ELBO values
    elbos = np.array([elbo.numpy() for elbo in elbos])
    dmgp.core.plot_elbo(elbos, log_dir=log_dir, show_plot=False)

    # Perform posterior inference
    def inference_DMGP(data, ids, split):
        print(f'\nRunning posterior inference for {split} data...')

        X, G, Y = data
        G_ids, X_ids = ids

        n = X.shape[0]
        p = G.shape[0]

        log_pred_density = 0.

        if n * p < 10000:
            f_means, f_vars = model.predict_f(data, ids, sep='fixed')
            f_mean_X_fixed, f_mean_X_random, f_mean_G_fixed, f_mean_G_random = f_means
            f_var_X_fixed, f_var_X_random, f_var_G_fixed, f_var_G_random = f_vars

            F_mean_X_fixed = dmgp.core.vec_inv(f_mean_X_fixed, (p,n))
            F_mean_X_random = dmgp.core.vec_inv(f_mean_X_random, (p,n))
            F_mean_X = F_mean_X_fixed + F_mean_X_random

            F_mean_G_fixed = dmgp.core.vec_inv(f_mean_G_fixed, (p,n))
            F_mean_G_random = dmgp.core.vec_inv(f_mean_G_random, (p,n))
            F_mean_G = F_mean_G_fixed + F_mean_G_random

            F_mean_fixed = F_mean_X_fixed + F_mean_G_fixed
            F_mean_random = F_mean_X_random + F_mean_G_random
            F_mean = F_mean_fixed + F_mean_random

            F_var_X_fixed = dmgp.core.vec_inv(tf.expand_dims(f_var_X_fixed, axis=-1), (p,n))
            F_var_X_random = dmgp.core.vec_inv(tf.expand_dims(f_var_X_random, axis=-1), (p,n))
            F_var_X = F_var_X_fixed + F_var_X_random

            F_var_G_fixed = dmgp.core.vec_inv(tf.expand_dims(f_var_G_fixed, axis=-1), (p,n))
            F_var_G_random = dmgp.core.vec_inv(tf.expand_dims(f_var_G_random, axis=-1), (p,n))
            F_var_G = F_var_G_fixed + F_var_G_random

            F_var_fixed = F_var_X_fixed + F_var_G_fixed
            F_var_random = F_var_X_random + F_var_G_random
            F_var = F_var_fixed + F_var_random

            y_mean, y_var = model.predict_y(data, ids)
            Y_mean = dmgp.core.vec_inv(y_mean, (p,n))
            Y_var = dmgp.core.vec_inv(y_var, (p,n))

            assert(F_mean.shape == F_var.shape == Y.shape)
            assert(Y_mean.shape == Y_var.shape == Y.shape)

            if 'nlpd' in acc_metrics:
                if split == 'missing':
                    log_pred_density = model.predict_log_density(data, ids, mask=~data_mask)
                else:
                    log_pred_density = model.predict_log_density(data, ids)
    
        # Default: Use mini-batch settings for training
        else:
            Y_mean = np.zeros((p,n), dtype=np.float64)
            Y_var = np.zeros((p,n), dtype=np.float64)

            F_mean_X_fixed = np.zeros((p,n), dtype=np.float64)
            F_mean_X_random = np.zeros((p,n), dtype=np.float64)
            F_var_X_fixed = np.zeros((p,n), dtype=np.float64)
            F_var_X_random = np.zeros((p,n), dtype=np.float64)

            F_mean_G_fixed = np.zeros((p,n), dtype=np.float64)
            F_mean_G_random = np.zeros((p,n), dtype=np.float64)
            F_var_G_fixed = np.zeros((p,n), dtype=np.float64)
            F_var_G_random = np.zeros((p,n), dtype=np.float64)

            idx_splits = get_idx_splits(np.array(range(p)),
                                        np.array(range(n)),
                                        b, a)

            for idx_split in tqdm(idx_splits):
                batch, batch_ids, mask_batch = get_batch(data, (np.array(range(p)), np.array(range(n))), idx_split, mask=data_mask)
                b_prime = batch[1].shape[0]
                a_prime = batch[0].shape[0]
                f_means_batch, f_vars_batch = model.predict_f(batch, batch_ids, sep='fixed')
                f_mean_X_fixed_batch, f_mean_X_random_batch, f_mean_G_fixed_batch, f_mean_G_random_batch = f_means_batch
                f_var_X_fixed_batch, f_var_X_random_batch, f_var_G_fixed_batch, f_var_G_random_batch = f_vars_batch

                y_mean_batch, y_var_batch = model.predict_y(batch, batch_ids)
                Y_mean_batch = dmgp.core.vec_inv(y_mean_batch, (b_prime,a_prime))
                Y_var_batch = dmgp.core.vec_inv(y_var_batch, (b_prime,a_prime))

                # Fill in patch
                ix = np.ix_(*idx_split)
                F_mean_X_fixed[ix] = dmgp.core.vec_inv(f_mean_X_fixed_batch, (b_prime,a_prime))
                F_mean_X_random[ix] = dmgp.core.vec_inv(f_mean_X_random_batch, (b_prime,a_prime))
                F_var_X_fixed[ix] = dmgp.core.vec_inv(f_var_X_fixed_batch, (b_prime,a_prime))
                F_var_X_random[ix] = dmgp.core.vec_inv(f_var_X_random_batch, (b_prime,a_prime))

                F_mean_G_fixed[ix] = dmgp.core.vec_inv(f_mean_G_fixed_batch, (b_prime,a_prime))
                F_mean_G_random[ix] = dmgp.core.vec_inv(f_mean_G_random_batch, (b_prime,a_prime))
                F_var_G_fixed[ix] = dmgp.core.vec_inv(f_var_G_fixed_batch, (b_prime,a_prime))
                F_var_G_random[ix] = dmgp.core.vec_inv(f_var_G_random_batch, (b_prime,a_prime))

                Y_mean[ix] = Y_mean_batch
                Y_var[ix] = Y_var_batch

                if 'nlpd' in acc_metrics:
                    if split == 'missing':
                        log_pred_density += model.predict_log_density(batch, batch_ids, mask=~mask_batch)
                    else:
                        log_pred_density += model.predict_log_density(batch, batch_ids)

            F_mean_fixed = F_mean_X_fixed + F_mean_G_fixed
            F_mean_random = F_mean_X_random + F_mean_G_random
            F_mean = F_mean_fixed + F_mean_random

            F_var_fixed = F_var_X_fixed + F_var_G_fixed
            F_var_random = F_var_X_random + F_var_G_random
            F_var = F_var_fixed + F_var_random

        npz_dict = {'F_mean_X_fixed': F_mean_X_fixed,
                    'F_mean_X_random': F_mean_X_random,
                    'F_var_X_fixed': F_var_X_fixed,
                    'F_var_X_random': F_var_X_random,
                    'F_mean_G_fixed': F_mean_G_fixed,
                    'F_var_G_fixed': F_var_G_fixed,
                    'F_mean_G_random': F_mean_G_random,
                    'F_var_G_random': F_var_G_random,
                    'F_mean_fixed': F_mean_fixed,
                    'F_var_fixed': F_var_fixed,
                    'F_mean_random': F_mean_random,
                    'F_var_random': F_var_random,
                    'F_mean': F_mean,
                    'F_var': F_var,
                    'Y_mean': Y_mean,
                    'Y_var': Y_var}

        # Save posterior
        np.savez(osp.join(log_dir, f'{split}_posterior.npz'), **npz_dict)

        # Calculate accuracy/error
        for acc_metric in acc_metrics:
            with open(osp.join(log_dir, f'{split}_{acc_metric}.txt'), 'w') as fh:
                if acc_metric == 'mse':
                    if split == 'missing':
                        mse = np.mean((Y[~data_mask] - Y_mean[~data_mask])**2)

                        fh.write(f'MSE ({split})={mse}')
                        print(f'MSE ({split})={mse}')

                    else:
                        mse = np.mean((Y - Y_mean)**2, axis=1)

                        for i in range(mse.shape[0]):
                            fh.write(f'Task {i+1}: MSE={mse[i]}\n')

                        fh.write(f'MSE ({split})={np.mean(mse, axis=0)}')
                        print(f'MSE ({split})={np.mean(mse, axis=0)}')

                elif acc_metric == 'rmse':
                    if split == 'missing':
                        rmse = np.sqrt(np.mean((Y[~data_mask] - Y_mean[~data_mask])**2))

                        fh.write(f'RMSE ({split})={rmse}')
                        print(f'RMSE ({split})={rmse}')

                    else:
                        rmse = np.sqrt(np.mean((Y - Y_mean)**2, axis=1))

                        for i in range(rmse.shape[0]):
                            fh.write(f'Task {i+1}: RMSE={rmse[i]}\n')

                        fh.write(f'RMSE ({split})={np.sqrt(np.mean(rmse**2, axis=0))}')
                        print(f'RMSE ({split})={np.sqrt(np.mean(rmse**2, axis=0))}')

                elif acc_metric == 'mae':
                    if split == 'missing':
                        mae = np.mean(np.abs(Y[~data_mask] - Y_mean[~data_mask]))

                        fh.write(f'MAE ({split})={mae}')
                        print(f'MAE ({split})={mae}')

                    else:
                        mae = np.mean(np.abs(Y - Y_mean), axis=1)

                        for i in range(mae.shape[0]):
                            fh.write(f'Task {i+1}: MAE={mae[i]}\n')

                        fh.write(f'MAE ({split})={np.mean(mae, axis=0)}')
                        print(f'MAE ({split})={np.mean(mae, axis=0)}')

                elif acc_metric == 'nlpd':
                    nlpd = -log_pred_density

                    if split == 'train' or split == 'test':
                        nlpd /= float(n * p)
                    elif split == 'missing':
                        nlpd /= float(n * p) - np.sum(data_mask.astype(np.int))

                    fh.write(f'NLPD ({split})={nlpd}')
                    print(f'NLPD ({split})={nlpd}')

    # Run posterior inference
    if miss_flag:
        inference_DMGP(train_data, train_ids, 'train')
        inference_DMGP(train_data, train_ids, 'missing')
    else:
        inference_DMGP(train_data, train_ids, 'train')
        inference_DMGP(test_data, test_ids, 'test')

    # Record train parameters in log directory
    with open(osp.join(log_dir, 'train_params.yaml'), 'w') as fh:
        yaml.dump(train_params, fh)

def main(**kwargs):
    n_exps = int(kwargs['n_exps'])
    miss_flag = not kwargs['test']
    show_summary = kwargs['show_summary']

    if n_exps == 0 or n_exps < -1:
        raise RuntimeError('[Error] Invalid number of experiments specified.')

    # Check model train logging directory exists
    main_log_dir = './logs/train_dmgp_logs'
    if not osp.isdir(main_log_dir):
        os.makedirs(main_log_dir)

    with open(TRAIN_PARAMS_PATH, 'r') as fh:
        train_params = yaml.load(fh, Loader=yaml.FullLoader)

    print(f'Loaded DMGP train parameter settings.')

    if n_exps == -1:
        train_dmgp(train_params, main_log_dir, show_summary=show_summary, miss_flag=miss_flag)
    else:
        for exp_id in range(1,n_exps+1):
            print(f'Running experiment {exp_id}...')
            train_dmgp(train_params, main_log_dir, show_summary=show_summary, exp_id=exp_id, miss_flag=miss_flag)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_exps', default=-1, help='Option to run multiple experiments in sequence')
    parser.add_argument('--test', default=False, help='Option to disable missing value prediction', action='store_true')
    parser.add_argument('--show_summary', default=False, help='Show model summary')
    args = parser.parse_args()

    main(**vars(args))
