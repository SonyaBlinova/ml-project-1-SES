import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from implementations import *
from utils import *
from plots import *
from proj1_helpers import *

def cross_validation_baseline(type_, y, x, k_indices, k, lambda_ = None, gamma = None, initial_w = None, max_iters = None, degree=0):
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
        initial_w = np.random.rand((x_train_poly.shape[1]))
        w, loss = least_squares_GD(y_train, x_train_poly, initial_w, max_iters, gamma, plot_loss = False)
        y_pred = predict_labels(w, x_test_poly)
        accuracy_  =  accuracy(y_pred, y_test)        
    elif type_ == 'SGD':
        initial_w = np.random.rand((x_train_poly.shape[1]))
        w, loss = least_squares_SGD(y_train, x_train_poly, initial_w, max_iters, gamma, plot_loss = False)
        y_pred = predict_labels(w, x_test_poly)
        accuracy_  =  accuracy(y_pred, y_test)  
    elif type_ == 'RR':
        w, loss = ridge_regression(y_train, x_train_poly, lambda_)
        y_pred = predict_labels(w, x_test_poly)
        accuracy_  =  accuracy(y_pred, y_test)  
    elif type_ == 'LR':
        initial_w = np.random.rand((x_train_poly.shape[1]))
        w, loss = logistic_regression(y_train, x_train_poly, initial_w, max_iters, gamma, plot_loss = False)
        y_pred = predict_labels_for_log(w, x_test_poly)
        accuracy_  =  accuracy(y_pred, y_test)  
    elif type_ == 'RLR':
        initial_w = np.random.rand((x_train_poly.shape[1]))
        w, loss = reg_logistic_regression(y_train, x_train_poly, lambda_, initial_w, max_iters, gamma, plot_loss = False)
        y_pred = predict_labels_for_log(w, x_test_poly)
        accuracy_  =  accuracy(y_pred, y_test)  
    else:
        raise TypeError(f"{type_} Wrong type!")
    return accuracy_

def cross_validation_demo_baseline(type_, y, tx, bd_left, bd_right, seed, gammas=None, max_iters=None, lambdas=None, degrees=None):
    k_fold = 3
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    global_acc = []
    
    if type_ in ['GD', 'SGD', 'LR']:
        params_1 = max_iters
        params_2 = gammas
        label_ = 'test error, num_iters = '
        xlabel = 'gamma'
    elif type_ in ['RR']:
        params_1 = degrees
        params_2 = lambdas
        label_ = 'test error, degree = '
        xlabel = 'lambda'
    elif type_ in ['RLR']:
        params_1 = degrees
        params_2 = lambdas
        label_ = 'test error, degree = '
        xlabel = 'lambda'
    else:
        raise NotImplementedError
        
    # cross validation
    for param_1 in tqdm(params_1):
        acc = []
        for param_2 in params_2:
            acc_batch = []
            for k in range(k_fold):
                if type_ in ['GD', 'SGD', 'LR']:
                    acc_ = cross_validation_baseline(type_=type_, y=y, x=tx, k_indices=k_indices, k=k, gamma = param_2, max_iters = param_1)
                elif type_ in ['RR']:
                    acc_ = cross_validation_baseline(type_=type_, y=y, x=tx, k_indices=k_indices, k=k, lambda_=param_2, degree=param_1)
                elif type_ in ['RLR']:
                    acc_ = cross_validation_baseline(type_=type_, y=y, x=tx, k_indices=k_indices, k=k, gamma = gammas, max_iters = max_iters, lambda_=param_2)
                else:
                    raise NotImplementedError
                acc_batch.append(acc_)
            acc.append(np.mean(acc_batch))
        global_acc.append(acc)

    print("Accuracy is {:.4f}".format(np.max(global_acc)))
    
    #plotting results
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(len(params_1)):
        label = label_ + str(params_1[i])
        if xlabel == 'gamma' and type_ != 'LR':
            plt.plot(params_2, global_acc[i], marker=".", label=label)
#             plt.semilogx(params_2, global_acc[i], marker=".", label=label)
#             plt.xlim(10**bd_left, 10**bd_right)
        else:
            plt.semilogx(params_2, global_acc[i], marker=".", label=label)
            plt.xlim(10**bd_left, 10**bd_right)
    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel("Accuracy", fontsize=17)
    plt.title("Cross validation for the " + type_, fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig('data/cross_val_basline'+type_+'.png')