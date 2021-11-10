#%%
import numpy as np
import random as r

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
from typing import Callable
import numpy as np


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
from typing import Callable
from typing import List
import numpy as np

# defined types
Array_function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_function]


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
