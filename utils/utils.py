import os
import errno
import random
import numpy as np
import torch

def createDir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    
def setSeed(seed):
    # Setup seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)