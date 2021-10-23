from implementations import *
from utils import *
from plots import *

def cross_validation(type_, y, x, k_indices, k, lambda_ = None, gamma = None, initial_w = None, max_iters = None, degree=0):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train
    not_k_indices = set(np.arange(len(x))) - set(k_indices[k])
    x_train = x[list(not_k_indices)]
    y_train = y[list(not_k_indices)]
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]

    # form data with polynomial degree
    if degree == 0:
        x_train_poly = x_train
        x_test_poly = x_test
    else:
        x_train_poly = build_poly(x_train, degree)
        x_test_poly = build_poly(x_test, degree)


    # calculate the loss for train and test data
    if type_ == 'GD':
        w, loss_tr = least_squares_GD(y_train, x_train_poly, initial_w, max_iters, gamma, plot_loss = False)
        loss_te = compute_mse(y_test, x_test_poly, w)
    elif type_ == 'SGD':
        w, loss_tr = least_squares_SGD(y_train, x_train_poly, initial_w, max_iters, gamma, plot_loss = False)
        loss_te = compute_mse(y_test, x_test_poly, w)
    elif type_ == 'RR':
        w, loss_tr = ridge_regression(y_train, x_train_poly, lambda_)
        loss_te = compute_mse(y_test, x_test_poly, w)
    elif type_ == 'LR':
        w, loss_tr = logistic_regression(y_train, x_train_poly, initial_w, max_iters, gamma, plot_loss = False)
        h = sigmoid(x_test_poly, w)        
        loss_te = - 1/y_test.shape[0]*np.sum((y_test == 1)*np.log(h) + (y_test == -1)*np.log(1 - h))
    elif type_ == 'RLR':
        w, loss_tr = reg_logistic_regression(y_train, x_train_poly, lambda_, initial_w, max_iters, gamma, plot_loss = True)
        h = sigmoid(x_test_poly, w)        
        loss_te = - 1/y_test.shape[0]*np.sum((y_test == 1)*np.log(h) + (y_test == -1)*np.log(1 - h))
    else:
        raise TypeError(f"{type_} Wrong type!")
    return loss_tr, loss_te