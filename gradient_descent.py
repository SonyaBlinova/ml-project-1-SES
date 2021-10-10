# -*- coding: utf-8 -*-
"""Gradient Descent"""
from costs import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # compute gradient and error vector
    # ***************************************************
    N = y.shape[0]
    error = y - tx @ w
    return (- (tx.T @ error) / N),error


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # compute gradient and loss
        # ***************************************************
        gradient, error = compute_gradient(y, tx, w)
        losss = loss(y, tx, w)
        # ***************************************************
        # update w by gradient
        # ***************************************************
        w = w - gamma * gradient
        #print(w.shape)
        # store w and loss
        ws.append(w)
        losses.append(losss)

    return losses, ws
