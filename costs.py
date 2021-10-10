import numpy as np
#the cost functions

def error(y,tx,w):
    return y - tx @ w

def loss(y,tx,w):
    e = error(y,tx,w)
    return np.mean(e**2) / 2

def mse_loss(error):
    return np.mean(error**2) / 2

def mae_loss(error):
    return np.mean(np.abs(error))

def rmse_loss(error):
    return np.sqrt(compute_mse_loss(error)*2)


