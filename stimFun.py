import numpy as np

def stimFun(M, T, stim_type):
    s = np.ones((M, T))
    """
    if stim_type == 1:
        s[5000:20000] = 0
    elif stim_type == 2:
        s[5000:20000] = 0
        s[20000:30000] = 0"""
    return s
