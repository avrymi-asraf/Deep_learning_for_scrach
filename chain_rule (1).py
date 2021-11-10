import numpy as np




#%%
def sigmoid(x:np.ndarray)-> np.ndarray:
    """Apply the sigmoid function to each element in the input ndarray"""
    return 1/(1+np.exp(-x))
