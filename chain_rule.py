# %%
from collections import namedtuple
import numpy as np
from typing import Callable
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from functions_dereative_nested import *


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
    if ind_chain == 0:
        f1_x = chain[0](input_range)
        return deriv(chain[0],input_range)*rection_der(chain,f1_x,ind_chain+1)
    if input_range == len(chain):
        fx = chain[ind_chain](input_range)
        return deriv(chain[ind_chain],input_range)*rection_der(chain,fx,ind_chain+1)
    





# %%



