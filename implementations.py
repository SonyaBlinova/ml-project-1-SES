import numpy as np
import matplotlib.pyplot as plt
from utils import *

def gradient_descent(df, y, tx, w_0, gamma, max_iter, return_all_steps = False):
    """
    Minimization using gradient descent algorithm.
    
    Parameters
    ----------
    df : function
        Function takes as input (y, tx, w_0) end return gradient vector.
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    w_0 : ndarray
        Initial weights.
    gamma : float
        Grandient descent step.
    max_iters : int
        Number of iteration.
    return_all_steps : bool, optional
        If argument is true, than gradient_descent returns all steps calculated during minimization.
        
    Returns
    -------
    w : ndarray
        Final weights.
    all_steps : list of ndarray, optional
        All steps calculated during minimization.
    """
    steps = [w_0.copy()]
    for _ in range(max_iter):
        w_0 = w_0 - gamma * df(y, tx, w_0)
        steps += [w_0.copy()]
        
    if return_all_steps:
        return w_0, steps
    return w_0

# def stochastic_gradient_descent(df, y, tx, w_0, gamma, max_iter, bath_size = 1, return_all_steps = False):
#     pass
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    initial_w : ndarray
        Initial weights.
    batch_size : int
        The size of a batch.
    max_iters : int
        Number of iteration.
    gamma : float
        Grandient descent step.
        
    Returns
    -------
    losses : list of ndarray
        All losses calculated during minimization.
    ws : list of ndarray
        All weights calculated during minimization.
    """
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad, loss = compute_stoch_gradient(y, tx, w)
        w = w - gamma*grad
        ws.append(w)
        losses.append(loss)
    return losses, ws

def ridge_regression(y, tx, lambda_):
    """
    Minimization using ridge regression.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    lambda_ : float
        Regularization parameter.
        
    Returns
    -------
    loss : float
        Minimization loss.
    w : ndarray
        Final weights.
    """
    
    N = tx.shape[1]
    w = np.linalg.solve(tx.T @ tx + 2*N*lambda_*np.eye(N), tx.T @ y)
    loss = compute_mse(y, tx, w)
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma, plot_loss = False):
    """
    Minimization using logistic regression.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    initial_w : ndarray
        Initial weights.
    max_iters : int
        Number of iteration.
    gamma : float
        Grandient descent step.
    plot_loss : bool, optional
        Clarification whether to draw a graph of changes in the loss function. Default False.
        
    Returns
    -------
    w : ndarray
        Final weights.
    final_loss : float
        Final minimization loss.
    """
    def compute_loss(y, w):
        h = sigmoid(tx, w)        
        loss = - 1/y.shape[0]*np.sum((y == 1)*np.log(h) + (y == -1)*np.log(1 - h))
        return loss 

    def df(y, tx, w):
        h = sigmoid(tx, w)
        return  1/y.shape[0]*((y == 1)*(1 - h) - (y == -1)*h)@tx
        
    w, steps = gradient_descent(df, y, tx, initial_w, gamma, max_iters, return_all_steps = True)
    
    if plot_loss:
        loss_info = []
        for step in steps:
            loss_info += [compute_loss(y, step)]
        
        plt.plot(loss_info)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
    return w, compute_loss(y, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, plot_loss = True):
    """
    Minimization using logistic regression with regularization.
    
    Parameters
    ----------
    y : np.array
        Target values belonging to the set {-1, 1}.
    tx : np.array
        Matrix of features.
    initial_w : np.array
        Initial weights.
    max_iters : int
        Number of iteration.
    gamma : float
        Grandient descent step.
    plot_loss : bool, optional
        Clarification whether to draw a graph of changes in the loss function. Default False.
        
    Returns
    -------
    w : ndarray
        Final weights.
    final_loss : float
        Final minimization loss.   
    """
    def compute_loss(y, w, lambda_ = 0):
        h = sigmoid(tx, w.T)
        loss = - 1/y.shape[0]*np.sum((y == 1)*np.log(h) + (y == -1)*np.log(1 - h))
        loss += lambda_*np.sum(w**2)
        return loss 
    
    def df(y, tx, w):
        h = sigmoid(tx, w)
        return  1/y.shape[0]*((y == 1)*(1 - h) - (y == -1)*h)@tx + 2*lambda_*w
        
    w, steps = gradient_descent(df, y, tx, initial_w, gamma, max_iters, return_all_steps = True)
    
    if plot_loss:
        loss_info = []
        for step in steps:
            loss_info += [compute_loss(y, step)]
        
        plt.plot(loss_info)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
    return w, compute_loss(y, w)
