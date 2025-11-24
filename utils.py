"""Utility functions: metrics and walk-forward cross validation helper.
""" 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(a,b): return np.sqrt(mean_squared_error(a.reshape(-1), b.reshape(-1)))
def mae(a,b): return mean_absolute_error(a.reshape(-1), b.reshape(-1))
def mape(a,b): return np.mean(np.abs((a-b)/ (np.clip(a,1e-6,None))))*100

def walk_forward_split(arr, input_window, horizon, step=1):
    L = arr.shape[0]
    idxs = []
    for start in range(0, L - input_window - horizon + 1, step):
        idxs.append((start, start+input_window, start+input_window, start+input_window+horizon))
    return idxs
