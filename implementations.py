import numpy as np
from proj1_helpers import *
from costs import *
from gradient_descent import *
from stochastic_gradient_descent import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    w = initial_w
    for n_iter in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        w = w - (gamma * gradient)
    loss = mse_loss(error)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma) : 
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient, error = compute_stoch_gradient(minibatch_y,minibatch_tx, w) #stoch_gradient same as gradient
            w = w - (gamma * gradient)
    loss = mse_loss(error)
    return w, loss

def least_squares(y,tx): 
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y,tx,lambda_):
    """implement ridge regression."""
    N=tx.shape[1]
    I=np.eye(N)
    w=np.linalg.solve(tx.T @ tx + 2*N*lambda_*I , tx.T @ y)
    loss = compute_loss_ridge(y,tx,w,lambda_)
    return w, loss

def logistic_regression(y,tx,initial_w,max_iters,gamma):
    raise NotImplementedError

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    raise NotImplementedError


