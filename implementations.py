import numpy as np
import matplotlib.pyplot as plt
from utils import *

def sigmoid(tx, w):
    """
    Sigmoid function
    
    Parameters
    ----------
    tx : ndarray
        Matrix of features.
    initial_w : ndarray
        Initial weights.
        
    Returns
    -------
    sigmod : float
        Value of the sigmoid function.
    """
    
    return 1/(1 + np.exp(tx@w.T))

def logistic_regression(y, tx, initial_w, max_iters, gamma, plot_loss = False):
    """
    Minimization using logistic regression.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the interval [0, 1].
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
        Target values belonging to the interval [0, 1].
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
