import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(df, y, tx, w_0, gamma, max_iter, return_all_steps = False):
    """
    Minimization using gradient descent algorithm.
    
    Parameters
    ----------
    df : function
        Function takes as input (y, tx, w_0) end return gradient vector.
    y : ndarray
        Target values belonging to the interval [0, 1].
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

def stochastic_gradient_descent(df, y, tx, w_0, gamma, max_iter, bath_size = 1, return_all_steps = False):
    pass


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    shuffled = np.random.permutation(np.c_[x, y])
    shuffled_x = shuffled[:,0:-1]
    shuffled_y = shuffled[:,-1:]
    split = int(ratio * shuffled_x.shape[0])
    training_x = shuffled_x[:split]
    training_y = shuffled_y[:split]
    test_x = shuffled_x[split:]
    test_y = shuffled_y[split:]
    return training_x, test_x, training_y, test_y


def accuracy(y_pred, y_test):
    """
    Returns the percentage of difference between two numpy arrays
    """
    diff = y_test - y_pred
    return len(diff[diff==0])/len(diff)