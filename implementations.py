import numpy as np
import matplotlib.pyplot as plt
import random
from utils import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma, plot_loss = False):
    """
    Linear regression using gradient descent.
    
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

    def df(y, tx, w):
        return (-1/y.shape[0])*np.dot(tx.T, y - np.dot(tx, w))
        
    w, steps = gradient_descent(df, y, tx, initial_w, gamma, max_iters, return_all_steps = True)
    
    if plot_loss:
        loss_info = []
        for step in steps:
            loss_info += [compute_mse(y, tx, step)]
        
        plt.plot(loss_info)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
    return w, compute_mse(y, tx, w)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, plot_loss = False):
    """
    Linear regression using stochastic gradient descent.
    
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

    def df(y, tx, w):
        idx = random.choice(range(w.shape[0]))
        grad_dgd = np.zeros(w.shape[0])
        grad_dgd[idx] = (-1/y.shape[0])*np.dot(tx.T[idx], y - np.dot(tx, w))
        return grad_dgd
        
    w, steps = gradient_descent(df, y, tx, initial_w, gamma, max_iters, return_all_steps = True)
    
    if plot_loss:
        loss_info = []
        for step in steps:
            loss_info += [compute_mse(y, tx, step)]
        
        plt.plot(loss_info)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
        
    Returns
    -------
    w : ndarray
        Final weights.
    loss : float
        Minimization loss.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


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
    w : ndarray
        Final weights.
    loss : float
        Minimization loss.
    """
    
    N = tx.shape[1]
    w = np.linalg.solve(tx.T @ tx + 2*N*lambda_*np.eye(N), tx.T @ y)
    return w, compute_mse(y, tx, w)

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
    def df(y, tx, w):
        h = sigmoid(tx, w)
        return  1/y.shape[0]*((y == 1)*(1 - h) - (y == -1)*h)@tx
        
    w, steps = gradient_descent(df, y, tx, initial_w, gamma, max_iters, return_all_steps = True)
    
    if plot_loss:
        loss_info = []
        for step in steps:
            loss_info += [compute_logistic_loss(y, tx, step)]
        
        plt.plot(loss_info)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, plot_loss = False):
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
    def df(y, tx, w):
        h = sigmoid(tx, w)
        return  1/y.shape[0]*((y == 1)*(1 - h) - (y == -1)*h)@tx + 2*lambda_*w
        
    w, steps = gradient_descent(df, y, tx, initial_w, gamma, max_iters, return_all_steps = True)
    
    if plot_loss:
        loss_info = []
        for step in steps:
            loss_info += [compute_logistic_loss(y, tx, step)]
        
        plt.plot(loss_info)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
    return w, compute_logistic_loss(y, tx, w)
