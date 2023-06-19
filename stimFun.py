import numpy as np

def stimFun(M, T, stim_type):
    s = np.zeros((M, T))
    
    if stim_type == 1:
        val = 0
        for i in range(50000):
            for j in range(M):
                s[j][i] = val
                val -=0.00005
        val = 0
        for i in range(50001, 75000):
            for j in range(M):
                s[j][i] = val
                val +=0.00005
        val = 0
    elif stim_type == 2:
        s[5000:20000] = 0
        s[20000:30000] = 0
    return s
