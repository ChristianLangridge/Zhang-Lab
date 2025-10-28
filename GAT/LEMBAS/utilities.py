"""
Helper functions for running and training the SignalingModel. 
"""
import os
import time
from typing import List

import torch
import numpy as np

def set_seeds(seed: int=888):
    """Sets random seeds for torch operations.

    Parameters
    ----------
    seed : int, optional
        seed value, by default 888
    """
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ.keys():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_cores(n_cores: int):
    """Set environmental variables to ensure core usage is limited to n_cores

    Parameters
    ----------
    n_cores : int
        number of cores to use
    """
    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

def get_lr(iter: int, max_iter: int, max_height: float = 1e-3, 
             start_height: float=1e-5, end_height: float=1e-5, 
             peak: int = 1000):
    """Calculates learning rate for a given iteration during training.

    Parameters
    ----------
    iter : int
        the current iteration
    max_iter : int
        the maximum number of training iterations
    max_height : float, optional
        tuning parameters for learning for the first 95% of iterations, by default 1e-3
    start_height : float, optional
        tuning parameter for learning rate before peak iterations, by default 1e-5
    end_height : float, optional
        tuning parameter for learning rate afer peak iterations, by default 1e-5
    peak : int, optional
        the first # of iterations to calculate lr on (should be less than 95% 
        of max_iter), by default 1000

    Returns
    -------
    lr : float
        the learning rate
    """

    phase_length = 0.95 * max_iter
    if iter<=peak:
        effective_iter = iter/peak
        lr = (max_height-start_height) * 0.5 * (np.cos(np.pi*(effective_iter+1))+1) + start_height
    elif iter<=phase_length:
        effective_iter = (iter-peak)/(phase_length-peak)
        lr = (max_height-end_height) * 0.5 * (np.cos(np.pi*(effective_iter+2))+1) + end_height
    else:
        lr = end_height
    return lr

def get_lr_grokking(iter: int, max_iter: int, max_height: float = 1e-3, 
             start_height: float=1e-5, end_height: float=1e-5, 
             peak: int = 1000):
    """Calculates learning rate for a given iteration during training.

    Parameters
    ----------
    iter : int
        the current iteration
    max_iter : int
        the maximum number of training iterations
    max_height : float, optional
        tuning parameters for learning for the first 95% of iterations, by default 1e-3
    start_height : float, optional
        tuning parameter for learning rate before peak iterations, by default 1e-5
    end_height : float, optional
        tuning parameter for learning rate afer peak iterations, by default 1e-5
    peak : int, optional
        the first # of iterations to calculate lr on (should be less than 95% 
        of max_iter), by default 1000

    Returns
    -------
    lr : float
        the learning rate
    """

    phase_length = 0.5 * max_iter
    if iter<=peak:
        effective_iter = iter/peak
        lr = (max_height-start_height) * 0.5 * (np.cos(np.pi*(effective_iter+1))+1) + start_height
    elif iter<=phase_length:
        effective_iter = (iter-peak)/(phase_length-peak)
        lr = (max_height-end_height) * 0.5 * (np.cos(np.pi*(effective_iter+2))+1) + end_height
    else:
        # Linear decrease towards 1e-6
        lr = end_height - (end_height - 1e-8) * ((iter - phase_length) / (max_iter - phase_length))
    return lr

def initialize_progress(max_iter: int):
    """Track various stats of the progress of training the model.

    Parameters
    ----------
    max_iter : int
        the maximum number of training iterations

    Returns
    -------
    stats : dict
        a dictionary of progress statistics
    """
    stats = {}
    stats['start_time'] = time.time()
    stats['end_time'] = 0
    stats['iter_time'] = np.nan*np.ones(max_iter)
    
    stats['loss_mean'] = np.nan*np.ones(max_iter)
    stats['loss_sigma'] = np.nan*np.ones(max_iter)
    stats['eig_mean'] = np.nan*np.ones(max_iter)
    stats['eig_sigma'] = np.nan*np.ones(max_iter)

    stats['test'] = np.nan*np.ones(max_iter)
    stats['learning_rate'] = np.nan*np.ones(max_iter)
    stats['violations'] = np.nan*np.ones(max_iter)

    return stats

def update_progress(stats : dict, iter: int, 
                  loss: List[float]=None, eig: List[float]=None, 
                  learning_rate: float=None, n_sign_mismatches: float=None):
    """Updates various stats of the progress of training the model.

    Parameters
    ----------
    stats : dict
        a dictionary of progress statistics
    iter : int
        the current training iteration
    loss : List[float], optional
        a list of the loss (excluding regularizations) up to `iter` , by default None
    eig : List[float], optional
        a list of the spectral_radius up to `iter` , by default None
    learning_rate : float, optional
        the model learning rate at `iter`, by default None
    n_sign_mismatches : float, optional
        the total number of sign mismatches at `iter`, 
        output of `SignalingModel.signaling_network.count_sign_mismatch()`, by default None

    Returns
    -------
    stats : dict
        updated dictionary of progress statistics
    """
    if loss != None:
        stats['loss_mean'][iter] = np.mean(np.array(loss))
        stats['loss_sigma'][iter] = np.std(np.array(loss))
    if learning_rate != None:
        stats['learning_rate'][iter] = learning_rate
    if n_sign_mismatches != None:
        stats['violations'][iter] = n_sign_mismatches
    
    stats['iter_time'][iter] = time.time()

    return stats

def print_stats(stats, iter):
    """Prints various stats of the progress of training the model.

    Parameters
    ----------
    stats : dict
        a dictionary of progress statistics
    iter : int
        the current training iteration
    """
    msg = 'i={:.0f}'.format(iter)
    if not np.isnan(stats['loss_mean'][iter]):
        msg += ', l={:.5f}'.format(stats['loss_mean'][iter])
    # if not np.isnan(stats['test'][iter]):
    #     msg += ', t={:.5f}'.format(stats['test'][iter])
    if not np.isnan(stats['eig_mean'][iter]):
        msg += ', s={:.3f}'.format(stats['eig_mean'][iter])
    if not np.isnan(stats['learning_rate'][iter]):
        msg += ', r={:.5f}'.format(stats['learning_rate'][iter])
    if not np.isnan(stats['violations'][iter]):
        msg += ', v={:.0f}'.format(stats['violations'][iter])
        
    print(msg)

def get_moving_average(values: np.array, n_steps: int):
    """Get the moving average of a tracked state across n_steps. Serves to smooth value. 

    Parameters
    ----------
    values : np.array
        values on which to get the moving average
    n_steps : int
        number of steps across which to get the moving average

    Returns
    -------
    moving_average : np.array
        the moving average across values
    """
    moving_average = np.zeros(values.shape)
    for i in range(values.shape[0]):
        start = np.max((i-np.ceil(n_steps/2), 0)).astype(int)
        stop = np.min((i+np.ceil(n_steps/2), values.shape[0])).astype(int)
        moving_average[i] = np.mean(values[start:stop])
    return moving_average