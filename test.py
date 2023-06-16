import numpy as np
import torch

b1 = np.array([0.5, 0 , 0.3])
print(np.transpose(b1, np.newaxis))
print(b1[:, None])