# -*- coding: utf-8 -*-
import numpy as np
from costs import *
from helpers import *
from gradient_descent import *
from stochastic_gradient_descent import *
from logistic_gradient_hessian import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        w = w - (gamma * gradient)
    loss = compute_mse_loss(error)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient, error = compute_stoch_gradient(minibatch_y, minibatch_tx, w) #stoch_gradient same as gradient
            w = w - (gamma * gradient)
    loss = compute_mse_loss(error)
    return w, loss

def least_squares(y, tx):
    w = np.linalg.inv(tx.T @ tx).dot(tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    transpose = tx.T
    N = tx.shape[1]
    w = np.linalg.solve((transpose @ tx) + 2*N*lambda_*np.eye((N)), transpose @ y)
    loss = compute_rmse_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w.reshape((initial_w.shape[0], 1))
    for n_iter in range(max_iters):
        loss = compute_log_loss(y, tx, w)
        gradient = compute_log_gradient(y, tx, w)
        #hessian = compute_log_hessian(y, tx, w)
        #w = w - gamma * np.linalg.inv(hessian) @ gradient
        w = w - gamma * gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_log_loss(y, tx, w) + lambda_ * np.linalg.norm(w)**2
        gradient = compute_log_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss
