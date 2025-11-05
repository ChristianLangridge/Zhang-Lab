"""
Train the signaling model.
"""
from typing import Dict, List, Union
import time

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm  # Ensure tqdm is imported

import LEMBAS.utilities as utils


LR_PARAMS = {'max_iter': 50, 'learning_rate': 2e-3}
OTHER_PARAMS = {'batch_size': 1, 'noise_level': 10, 'gradient_noise_level': 1e-9}
REGULARIZATION_PARAMS = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                   'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5}
SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 10}
HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS}

def split_data(X_in: torch.Tensor, 
               y_out: torch.Tensor, 
               train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None}, 
              seed: int = 888):
    """Splits the data into train, test, and validation.

    Parameters
    ----------
    X_in : torch.Tensor
        input ligand concentrations. Index represents samples and columns represent a ligand. Values represent amount of ligand introduced (e.g., concentration). 
    y_out : torch.Tensor
        output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF.
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    seed : int, optional
        seed value, by default 888
    """
    
    if not np.isclose(sum([v for v in train_split_frac.values() if v]), 1):
        raise ValueError('Train-test-validation split must sum to 1')
    
    if not train_split_frac['validation'] or train_split_frac['validation'] == 0:
        X_train, X_test, y_train, y_test = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed)
        X_val, y_val = None, None
    else:
        X_train, _X, y_train, _y = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(_X, 
                                                    _y, 
                                                    train_size=train_split_frac['test']/(train_split_frac['test'] + train_split_frac['validation']),
                                                    random_state=seed)

    return X_train, X_test, X_val, y_train, y_test, y_val

class ModelData(Dataset):
    def __init__(self, X_in, y_out):
        self.X_in = X_in
        self.y_out = y_out
    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.X_in.shape[0]
    def __getitem__(self, idx: int):
        "Returns one sample of data, data and label (X, y)."
        return self.X_in[idx, :], self.y_out[idx, :]

def train_signaling_model(mod,  
                          optimizer: torch.optim, 
                          loss_fn: torch.nn.modules.loss,
                          reset_epoch: int = 200,
                          hyper_params: Dict[str, Union[int, float]] = None,
                          train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                          train_seed: int = None,
                          verbose: bool = True,
                          dict_to_bench: Dict[str, int] = {"name": "standard", "uniform": 0, "L2_or_rand": 0, "device": "cuda"}):
    """Trains the signaling model

    Parameters
    ----------
    mod : SignalingModel
        initialized signaling model. Suggested to also run `mod.signaling_network.prescale_weights` prior to training
    optimizer : torch.optim.adam.Adam
        optimizer to use during training
    loss_fn : torch.nn.modules.loss.MSELoss
        loss function to use during training
    reset_epoch : int, optional
        number of epochs upon which to reset the optimizer state, by default 200
    hyper_params : Dict[str, Union[int, float]], optional
        various hyper parameter inputs for training
            - 'max_iter' : the number of epochs, by default 5000
            - 'learning_rate' : the starting learning rate, by default 2e-3
            - 'batch_size' : number of samples per batch, by default 8
            - 'noise_level' : noise added to signaling network input, by default 10. Set to 0 for no noise. Makes model more robust. 
            - 'gradient_noise_level' : noise added to gradient after backward pass. Makes model more robust. 
            - 'reset_epoch' : number of epochs upon which to reset the optimizer state, by default 200
            - 'param_lambda_L2' : L2 regularization penalty term for most of the model weights and biases
            - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
            - 'ligand_lambda_L2' : L2 regularization penalty term for ligand biases
            - 'uniform_lambda_L2' : L2 regularization penalty term for 
            - 'uniform_max' : 
            - 'spectral_loss_factor' : regularization penalty term for 
            - 'n_probes_spectral' : 
            - 'power_steps_spectral' : 
            - 'subset_n_spectral' : 
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    train_seed : int, optional
        seed value, by default mod.seed. By explicitly making this an argument, it allows different train-test splits even 
        with the same mod.seed, e.g., for cross-validation
    verbose : bool, optional
        whether to print various progress stats across training epochs

    Returns
    -------
    mod : SignalingModel
        a copy of the input model with trained parameters
    cur_loss : List[float], optional
        a list of the loss (excluding regularizations) across training iterations
    cur_eig : List[float], optional
        a list of the spectral_radius across training iterations
    mean_loss : torch.Tensor
        mean TF activity loss across samples (independent of training)
    X_train : torch.Tensor
        the train split of the input data
    X_test : torch.Tensor
        the test split of the input data
    X_val : torch.Tensor
        the validation split of the input data
    y_train : torch.Tensor
        the train split of the output data
    y_test : torch.Tensor
        the test split of the input data
    y_val : torch.Tensor
        the validation split of the input data
    """
    if not hyper_params:
        hyper_params = HYPER_PARAMS.copy()
    else:
        hyper_params = {k: v for k, v in {**HYPER_PARAMS, **hyper_params}.items() if k in HYPER_PARAMS}  # give user input priority

    stats = utils.initialize_progress(hyper_params['max_iter'])

    mod = mod.copy()  # do not overwrite input
    optimizer = optimizer(mod.parameters(), lr=1)  # initial lr will be updated each iteration
    reset_state = optimizer.state.copy()

    X_in = mod.X_in
    y_out = mod.y_out
    mean_loss = loss_fn(torch.mean(y_out, dim=0) * torch.ones(y_out.shape, device=y_out.device), y_out)  # mean TF (across samples) loss
    print(mean_loss)
    # set up data objects

    if not train_seed:
        train_seed = mod.seed

    if train_split_frac['train'] == 1 or len(set([train_split_frac['test'], train_split_frac['validation']]).difference([None, 0])) == 0:
        X_train, X_test, X_val, y_train, y_test, y_val = X_in, None, None, y_out, None, None
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = split_data(X_in, y_out, train_split_frac, train_seed)

    train_data = ModelData(X_train.to('cpu'), y_train.to('cpu'))
    if mod.device == 'cuda':
        print('cuda')
        pin_memory = True
    else:
        print('not cuda')
        pin_memory = False

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=hyper_params['batch_size'],
                                  drop_last=False,
                                  pin_memory=pin_memory,
                                  shuffle=True)
    start_time = time.time()

    # begin iteration
   # mod.signaling_network.weights.data.masked_fill_(mask=mod.signaling_network.mask, value=0.0)  # fill non-interacting edges with 0

    for e in tqdm(range(hyper_params['max_iter']), desc='training'):
        # set learning rate
        cur_lr = utils.get_lr(e, hyper_params['max_iter'], max_height=hyper_params['learning_rate'],
                                start_height=hyper_params['learning_rate'] / 10, end_height=1e-6, peak=1000)
        optimizer.param_groups[0]['lr'] = cur_lr

        cur_loss = []
        cur_eig = []

        # iterate through batches
        if mod.seed:
            utils.set_seeds(mod.seed + e)
        for batch, (X_in_, y_out_) in enumerate(train_dataloader):
            optimizer.zero_grad()

            X_in_, y_out_ = X_in_.to(mod.device), y_out_.to(mod.device)

            # forward pass
        #    print('X_in_', X_in_.shape)
            X_full = mod.input_layer(X_in_)  # transform to full network with ligand input concentrations
            utils.set_seeds(mod.seed + mod._gradient_seed_counter)
            network_noise = torch.randn(X_full.shape, device=X_full.device)
            X_full = X_full + (hyper_params['noise_level'] * cur_lr * network_noise)  # add noise to make model robust
            Y_full = mod.signaling_network(X_full)  # train signaling network weights

            Y_hat = mod.output_layer(Y_full)

            # get prediction loss
            fit_loss = loss_fn(y_out_, Y_hat)

            # get regularization losses
         #   sign_reg = mod.signaling_network.sign_regularization(lambda_L1=hyper_params['moa_lambda_L1'])
        ##    ligand_reg = mod.ligand_regularization(lambda_L2=hyper_params['ligand_lambda_L2'])
        #    stability_loss, spectral_radius = mod.signaling_network.get_SS_loss(
        #        Y_full=Y_full.detach(), 
        #        spectral_loss_factor=hyper_params['spectral_loss_factor'],
        #        subset_n=hyper_params['subset_n_spectral'], 
        #        n_probes=hyper_params['n_probes_spectral'], 
        #        power_steps=hyper_params['power_steps_spectral']
        #    )
            uniform_reg = mod.uniform_regularization(lambda_L2=hyper_params['uniform_lambda_L2'] * cur_lr, 
                                                     Y_full=Y_full, 
                                                     target_min=0, 
                                                     target_max=hyper_params['uniform_max'])
            #param_reg = mod.L2_reg(hyper_params['param_lambda_L2'])

            total_loss = fit_loss   + uniform_reg #+ stability_loss

            total_loss.backward()

           #mod.add_gradient_noise(noise_level=hyper_params['gradient_noise_level'])

            optimizer.step()

    
            # store current batch metrics
            cur_loss.append(fit_loss.item())
          #  cur_eig.append('0')

        stats = utils.update_progress(stats, iter=e, loss=cur_loss, learning_rate=cur_lr, 
                                      n_sign_mismatches=mod.signaling_network.count_sign_mismatch())

        # print loss and weight
        if verbose:
            mean_loss_iter = np.mean(cur_loss)
            print(f"Iteration {e}: Mean Loss = {mean_loss_iter:.4f}")

        if np.logical_and(e % reset_epoch == 0, e > 0):
            optimizer.state = reset_state.copy()

    if verbose:
        mins, secs = divmod(time.time() - start_time, 60)
        print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

    X_val = 0
    y_val = 0

    return mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_val, y_train, y_test, y_val, train_dataloader

