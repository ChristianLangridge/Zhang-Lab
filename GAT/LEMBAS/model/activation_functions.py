"""
Defines various activations functions to be used in neural net.
"""

import numpy as np
import torch

from typing import Union

######## MML_ activation
def MML_activation(x: torch.Tensor, leak: Union[float, int]):
    """Returns the output of the Michaelis-Menten function

    Parameters
    ----------
    x : torch.Tensor
        a vector of input values
    leak : Union[float, int], optional
        parameter to tune extent of leaking, analogous to leaky ReLU

    Returns
    -------
    fx : torch.Tensor
        a vector of output values
    """
    fx = torch.nn.functional.leaky_relu(input = x, negative_slope = leak, inplace = False)
    shifted_x = 0.5 * (fx - 0.5)
    mask = torch.lt(shifted_x, 0.0)
    gated_x = fx + 10 * mask #prevents division by 0 issue on next line
    right_values = 0.5 + torch.div(shifted_x, gated_x)
    fx = mask * (fx - right_values) + right_values #-fx trick from relu
    return fx

def MML_delta_activation(x: torch.Tensor, leak: Union[float, int]):
    """Returns the derivative of the Michaelis-Menten function

    Parameters
    ----------
    x : torch.Tensor
        a vector of input values
    leak : Union[float, int], optional
        parameter to tune extent of leaking, analogous to leaky ReLU

    Returns
    -------
    y : torch.Tensor
        a vector of output derivative values
    """
    mask1 = x.le(0)
    y = torch.ones(x.shape, dtype = x.dtype, device = x.device) #derivative = 1 if nothing else is stated
    
    mask2 = x.gt(0.5)
    right_values = 0.25/torch.pow(x + 1e-12,2) - 1 # add psuedocount bc x = 0 will creat NaN
    y = y + mask2 * right_values
    y = y - (1-leak) * mask1
    return y

def MML_onestepdelta_activation_factor(Y_full: torch.Tensor, leak: Union[float, int]=0.01):
    """Adjusts weights for linearization in the spectral radius. 

    Note that this will only work for monotonic functions

    Parameters
    ----------
    Y_full : torch.Tensor
        _description_
    leak : Union[float, int], optional
        parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01

    Returns
    -------
    y : torch.Tensor
        _description_
    """
    
    y = torch.ones_like(Y_full)
    piece1 = Y_full.le(0)
    piece3 = Y_full.gt(0.5)

    safe_x = torch.clamp(1-Y_full, max=0.9999)
    right_values = 4 * torch.pow(safe_x, 2) - 1
    y = y + piece3 * right_values
    y = y - (1-leak) * piece1  
    return y

def MML_inv_activation(x, leak=0.01):
    safe_x = torch.clamp(1-x, max=0.9999)
    factor = 1/leak
    if leak>0:
        mask = x.lt(0)
        x = x + mask * (factor * x -x)
    #Else if it is zero it will be multiplied with a zero later so no need to cover this case
    mask = x.gt(0.5)
    right_values = 0.25/(safe_x) - x    
    x = x + right_values * mask
    return fx



######## leaky ReLU_ activation
def leakyReLU_activation(x, leak=0.01):
    """Returns the output of the leaky ReLU function

    Parameters
    ----------
    x : Iterable[Union[float, int]]
        a vector of input values
    leak : Union[float, int], optional
        parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01

    Returns
    -------
    fx
        a vector of output values
    """
    fx = np.copy(x)
    fx = np.where(fx < 0, fx * leak, fx)
    return fx

def leakyReLU_delta_activation(x, leak=0.01):
    """Returns the derivative of the leaky ReLU function

    Parameters
    ----------
    x : Iterable[Union[float, int]]
        a vector of input values
    leak : Union[float, int], optional
        parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01

    Returns
    -------
    y : Iterable[Union[float, int]]
        a vector of output derivative values
    """
    y = np.ones(x.shape) #derivative = 1 if nothing else is stated
    y = np.where(x <= 0, leak, y)  #let derivative be 0.01 at x=0
    return y

def leakyReLU_onestepdelta_activation_factor(yhatFull, leak=0.01):  #Note that this will only work for monoton functions
    y = torch.ones(y_hat_full.shape, dtype=y_hat_full.dtype)
    piece1 = y_hat_full<=0
    y[piece1] = torch.tensor(leak, dtype=y_hat_full.dtype)
    return y

def leakyReLU_inv_activation(x, leak=0.01):
    fx = np.copy(x)
    if leak>0:
        fx = np.where(fx < 0, fx/leak, fx)
    else:
        fx = np.where(fx < 0, 0, fx)
    return fx



######## sigmoid_ activation
def sigmoid_activation(x, leak=0):
    """Returns the output of the sigmoid function

    Parameters
    ----------
    x : Iterable[Union[float, int]]
        a vector of input values
    leak : Union[float, int], optional
        parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01

    Returns
    -------
    fx
        a vector of output values
    """
    fx = np.copy(x)
    #leak is not used for sigmoid_
    fx = 1/(1 + np.exp(-fx))
    return fx

def sigmoid_delta_activation(x, leak=0):
    """Returns the derivative of the sigmoid function

    Parameters
    ----------
    x : Iterable[Union[float, int]]
        a vector of input values
    leak : Union[float, int], optional
        parameter to tune extent of leaking, analogous to leaky ReLU, by default 0.01

    Returns
    -------
    y : Iterable[Union[float, int]]
        a vector of output derivative values
    """
    y = sigmoid_activation(x) * (1 - sigmoid_activation(x))
    return y

def sigmoid_onestepdelta_activation_factor(yhatFull, leak=0):  #Note that this will only work for monoton functions
    y = y_hat_full * (1- y_hat_full)
    return y

activation_function_map = {'MML': {'activation': MML_activation, 'delta': MML_delta_activation, 'onestepdelta': MML_onestepdelta_activation_factor},
                          'leaky_relu': {'activation': leakyReLU_activation, 'delta': leakyReLU_delta_activation, 'onestepdelta': leakyReLU_onestepdelta_activation_factor},
                           'sigmoid': {'activation': sigmoid_activation, 'delta': sigmoid_delta_activation, 'onestepdelta': sigmoid_onestepdelta_activation_factor}
                          }