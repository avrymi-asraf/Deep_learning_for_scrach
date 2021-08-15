# %%
from collections import namedtuple
import numpy as np
from typing import Callable
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from functions_dereative_nested import *


a = np.array([r.randint(-10, 10) for i in range(20)])
b = np.array([r.randint(-10, 10) for i in range(20)])
a * b

tow_d = np.array([a, b])
tow_d.sum(axis=0)
tow_d.sum(axis=1)


def square(x: np.ndarray) -> np.ndarray:
    """square every elemnet in ndarrauy"""
    return np.power(x, 2)


def leaky_relu(x: np.ndarray) -> np.ndarray:
    """Apply "Leaky ReLU" function to each element in ndarray"""
    return np.maximum(0.2 * x, x)


#%%
# defined types
Array_function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_function]




def deriv(
    func: Callable[[np.ndarray], np.ndarray], input_: np.ndarray, delta: float = 0.001
) -> np.ndarray:
    """return the deriative of function of every point in input array

    Args:
        func (Callable[[np.ndarray], np.ndarray]): function exspres by ndarray
        input (np.ndarray): point to calculte the derivative
        delta (float, optional): value of delte for moast function is enofe. Defaults to 0.001.

    Returns:
        np.ndarray: the derivative in points
    """
    return (func(input_ + delta) - func(input_)) / (2 * delta)


#%%

def chain_length_2(chain: Chain, a: np.ndarray) -> np.ndarray:
    """chain two functins

    Args:
        chain (Chain): list of ndarray functions
        a (np.ndarray): parameter to pass in functions

    Returns:
        np.ndarray:
    """
    assert len(chain) == 2, "The chain shuld be 2"
    f1 = chain[0]
    f2 = chain[1]
    return f1(f2(a))

# defined types
Array_function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_function]


# %%
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid function to each element in the input ndarray"""
    return 1/(1+np.exp(-x))


def chain_deriv_2(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    """Uses the chain rule to coumput the derivative two nested function

    Args:
        chain (Chain): the two functions
        input_range (np.ndarray): array to comput the derivative there

    Returns:
        np.ndarray: the deriative of all points
    """

    assert len(chain) == 2,\
        'the chain requires two functions'

    assert input_range.ndim == 1,\
        'function requires input range as 1 demonical'

    f1 = chain[0]
    f2 = chain[1]

    f1_of_x = f1(input_range)
    d_f1_dx = deriv(f1, input_range)

    d_d2_du = deriv(f2, f1_of_x)

    return d_f1_dx * d_d2_du

# %% Seaborn


PLOT_RANGE = np.arange(-3, 3, 0.01)
def chain(d): return chain_length_2([square, sigmoid], d)
def chain_der(d): return chain_deriv_2([square, sigmoid], d)


sns.set_context("notebook", rc={"lines.linewidth": 2})

sns.lineplot(x=PLOT_RANGE, y=chain(PLOT_RANGE))
sns.lineplot(x=PLOT_RANGE, y=chain_der(PLOT_RANGE))


# %%
def chain_deriv_3(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    """calcult deriv for 3 functions 

    Args:
        chain (Chain): 3 functions from numpy format
        input_range (np.ndarray): range to calcult, by nmpy range

    Returns:
        np.ndarray: numpy range 
    """

    assert len(chain) == 3, \
        'this function require chain to 3 functions'

    f1, f2,f3 = chain[0],chain[1],chain[2]


    #f1(x)
    f1_x = f1(input_range)

    #f2(f1(x))
    f2_x = f2(f1_x)

    #df3du 
    df3du = deriv(f3,f2_x)
    
    #df2du
    df2du = deriv(f2,f2_x) 
    
    #df1dx
    df1dx = deriv(f1,input_range)


    return df1dx*df2du*df3du



def rection_der(chain: Chain, input_range: np.ndarray,ind_chain:int) -> np.ndarray:
    """calcult derivative of chain function 

    Args:
        chain (Chain):list of function
        input_range (np.ndarray): input range to calcult 
        ind_chain (int): index of function

    Returns:
        np.ndarray: derivative of 
    """
    if ind_chain == len(chain)-1:
        f_x = chain[ind_chain](input_range)
        return deriv(chain[ind_chain],input_range)
    f_x = chain[ind_chain](input_range)
    return deriv(chain[ind_chain],input_range)*rection_der(chain,f_x,ind_chain+1,)
    


# %% plot 
chain_of_3 = [leaky_relu,square,sigmoid]
x = PLOT_RANGE
y = chain_deriv_3(chain_of_3,PLOT_RANGE)
sns.lineplot(x,y)



# %%
chain_of_3 = [leaky_relu,square,sigmoid]
x = PLOT_RANGE
y = chain_deriv_3(chain_of_3,PLOT_RANGE)
rection_der(chain_of_3,PLOT_RANGE,0)
sns.lineplot(x,y)
