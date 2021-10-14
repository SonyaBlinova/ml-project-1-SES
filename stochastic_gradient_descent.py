# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # implement stochastic gradient computation. It's same as the gradient descent.
    # ***************************************************
    N = y.shape[0]
    error = y - tx @ w
    return (- (tx.T @ error) / N), error

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # implement stochastic gradient descent.
    # ***************************************************
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # generate mini-batch
        # ***************************************************
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # ***************************************************
            # compute gradient and loss
            # ***************************************************
            gradient, _ = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            # ***************************************************
            # update w by gradient
            # ***************************************************
            w = w - gamma * gradient
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws