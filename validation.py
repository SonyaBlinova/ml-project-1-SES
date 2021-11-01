import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from implementations import *
from utils import *
from plots import *
from proj1_helpers import *

def cross_validation(type_, y, x, k_indices, k, lambda_ = None, gamma = None, max_iters = None, degree=0):
    """
    Cross validation for fixed parameters for 1 fold.
    Using for data with preprocess.
    
    Parameters:
    -----------
    type_ : str
        The type of model that is being cross-validated.
        type_ = {'GD', 'SGD', 'RR', 'LR', 'RLR'}.
    y : list
        List of 4 ndarrays of target values belonging to the set {-1, 1}.
    x : list
        List of 4 ndarrays of features.
    k_indices : list
        Contains k lists of test samples indices
    k : int
        Number of fold
    lambda_ : float, optional
        Regularization parameter.
        Default lambda_ = None.
    gamma : float, optoinal
        Step size of the gradient decent.
        Default gamma = None.
    max_iters : int, optional
        Maximum number of iteration.
        Default max_iters = None.
    degree : int, optional
        Polynomial degree of the features.
        If degree is equal to 0, polynomical transformation isn't applied to the features.
        Default degree = 0.
    
    Returns:
    --------
    accuracy_ : float
        Accuracy of the predicted labels.
    """
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    
    for i, (x_, y_) in enumerate(zip(x, y)):
        # get k'th subgroup in test, others in train
        not_k_indices = set(np.arange(len(x_))) - set(k_indices[i][k])
        x_train = x_[list(not_k_indices)]
        y_train = y_[list(not_k_indices)]
        x_test = x_[k_indices[i][k]]
        y_test = y_[k_indices[i][k]]

        # form data with polynomial degree
        if degree == 0:
            x_train_poly = x_train
            x_test_poly = x_test
        else:
            x_train_poly = build_poly(x_train, degree)
            x_test_poly = build_poly(x_test, degree)
            
        x_trains.append(x_train_poly)
        x_tests.append(x_test_poly)
        y_trains.append(y_train)
        y_tests.append(y_test)


    # calculate accuracy
    if type_ == 'GD':
        y_preds = []
        for i in range(4):
            initial_w = np.random.rand((x_trains[i].shape[1]))
            w, loss = least_squares_GD(y_trains[i], x_trains[i], initial_w, max_iters, gamma, plot_loss = False)
            y_pred = predict_labels(w, x_tests[i])
            y_preds.append(y_pred)
        accuracy_  =  accuracy(np.array(np.concatenate(y_preds)), np.array(np.concatenate(y_tests)))        
    elif type_ == 'SGD':
        y_preds = []
        for i in range(4):
            initial_w = np.random.rand((x_trains[i].shape[1]))
            w, loss = least_squares_SGD(y_trains[i], x_trains[i], initial_w, max_iters, gamma, plot_loss = False)
            y_pred = predict_labels(w, x_tests[i])
            y_preds.append(y_pred)
        accuracy_  =  accuracy(np.array(np.concatenate(y_preds)), np.array(np.concatenate(y_tests)))  
    elif type_ == 'RR':
        y_preds = []
        for i in range(4):
            w, loss = ridge_regression(y_trains[i], x_trains[i], lambda_)
            y_pred = predict_labels(w, x_tests[i])
            y_preds.append(y_pred)
        accuracy_  =  accuracy(np.array(np.concatenate(y_preds)), np.array(np.concatenate(y_tests)))  
    elif type_ == 'LR':
        y_preds = []
        for i in range(4):
            initial_w = np.random.rand((x_trains[i].shape[1]))
            w, loss = logistic_regression(y_trains[i], x_trains[i], initial_w, max_iters, gamma, plot_loss = False)
            y_pred = predict_labels_for_log(w, x_tests[i])
            y_preds.append(y_pred)
        accuracy_  =  accuracy(np.array(np.concatenate(y_preds)), np.array(np.concatenate(y_tests)))  
    elif type_ == 'RLR':
        y_preds = []
        for i in range(4):
            initial_w = np.random.rand((x_trains[i].shape[1]))
            w, loss = reg_logistic_regression(y_trains[i], x_trains[i], lambda_, initial_w, max_iters, gamma, plot_loss = False)
            y_pred = predict_labels_for_log(w, x_tests[i])
            y_preds.append(y_pred)
        accuracy_  =  accuracy(np.array(np.concatenate(y_preds)), np.array(np.concatenate(y_tests)))
    else:
        raise TypeError(f"{type_} Wrong type!")
    return accuracy_

def cross_validation_demo(type_, y, tx, bd_left, bd_right, seed, gammas=None, max_iters=None, lambdas=None, degrees=None):
    """
    Cross validation.
    Using for data with preprocess.
    
    Parameters:
    -----------
    type_ : str
        The type of model that is being cross-validated.
        type_ = {'GD', 'SGD', 'RR', 'LR', 'RLR'}.
    y : list
        List of 4 ndarrays of target values belonging to the set {-1, 1}.
    tx : list
        List of 4 ndarrays of features.
    bd_left : int
        Degree of 10. Left border of the parameter for cross validation.
    bd_right : int
        Degree of 10. Right border of the parameter for cross validation.
    seed : int
        The number used to initialize a pseudorandom number generator.
    gammas : list, optoinal
        List of step sizes of the gradient decent.
        Default gamma = None.
    max_iters : list, optional
        List of maximum numbers of iteration.
        Default max_iters = None.
    lambda_ : list, optional
        List of regularization parameters.
        Default lambda_ = None.
    degree : list, optional
        List of polynomial degrees of the features.
        If degree is equal to 0, polynomical transformation isn't applied to the features.
        Default degree = None.
    
    Returns:
    --------
    Plot describing the accuracy for different parameters
    """
    
    np.random.seed(seed)
    
    k_fold = 3
    # split data in k fold
    k_indices = []
    for i in range(4):
        k_indices.append(build_k_indices(y[i], k_fold, seed))
    # define lists to store the loss of training data and test data
    global_acc = []
    
    if type_ in ['GD', 'SGD', 'LR']:
        params_1 = max_iters
        params_2 = gammas
        label_ = 'max_iters = '
        xlabel = 'Gamma'
    elif type_ in ['RR']:
        params_1 = degrees
        params_2 = lambdas
        label_ = 'degree = '
        xlabel = 'Lambda'
    elif type_ in ['RLR']:
        params_1 = degrees
        params_2 = lambdas
        label_ = 'degree = '
        xlabel = 'Lambda'
    else:
        raise NotImplementedError
        
    # cross validation
    for param_1 in tqdm(params_1):
        acc = []
        for param_2 in params_2:
            acc_batch = []
            for k in range(k_fold):
                if type_ in ['GD', 'SGD', 'LR']:
                    acc_ = cross_validation(type_=type_, y=y, x=tx, k_indices=k_indices, k=k, gamma = param_2, max_iters = param_1)
                elif type_ in ['RR']:
                    acc_ = cross_validation(type_=type_, y=y, x=tx, k_indices=k_indices, k=k, lambda_=param_2, degree=param_1)
                elif type_ in ['RLR']:
                    acc_ = cross_validation(type_=type_, y=y, x=tx, k_indices=k_indices, k=k, gamma = gammas, max_iters = max_iters, lambda_=param_2, degree=param_1)
                else:
                    raise TypeError(f"{type_} Wrong type!")
                acc_batch.append(acc_)
            acc.append(np.mean(acc_batch))
        global_acc.append(acc)

    print("Accuracy is {:.4f}".format(np.max(global_acc)))
    
    #plotting results
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(params_1)):
        label = label_ + str(params_1[i])
        if xlabel == 'Gamma' and type_ != 'LR':
            plt.plot(params_2, global_acc[i], marker=".", label=label)
        else:
            plt.semilogx(params_2, global_acc[i], marker=".", label=label)
            plt.xlim(10**bd_left, 10**bd_right)
    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel("Accuracy", fontsize=17)
    plt.title("Cross validation for the " + type_, fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig('cross_validation'+type_+'.pdf')
    
    
def cross_validation_baseline(type_, y, x, k_indices, k, lambda_ = None, gamma = None, max_iters = None, degree=0):
    """
    Cross validation for fixed parameters for 1 fold.
    Using for data without preprocess.
    
    Parameters:
    -----------
    type_ : str
        The type of model that is being cross-validated.
        type_ = {'GD', 'SGD', 'RR', 'LR', 'RLR'}.
    y : ndarray
        Target values belonging to the set {-1, 1}.
    x : ndarray
        Matrix of features.
    k_indices : list
        Contains k lists of test samples indices
    k : int
        Number of fold
    lambda_ : float, optional
        Regularization parameter.
        Default lambda_ = None.
    gamma : float, optoinal
        Step size of the gradient decent.
        Default gamma = None.
    max_iters : int, optional
        Maximum number of iteration.
        Default max_iters = None.
    degree : int, optional
        Polynomial degree of the features.
        If degree is equal to 0, polynomical transformation isn't applied to the features.
        Default degree = 0.
    
    Returns:
    --------
    accuracy_ : float
        Accuracy of the predicted labels.
    """
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


    # calculate accuracy
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
    """
    Cross validation.
    Using for data without preprocess.
    
    Parameters:
    -----------
    type_ : str
        The type of model that is being cross-validated.
        type_ = {'GD', 'SGD', 'RR', 'LR', 'RLR'}.
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    bd_left : int
        Degree of 10. Left border of the parameter for cross validation.
    bd_right : int
        Degree of 10. Right border of the parameter for cross validation.
    seed : int
        The number used to initialize a pseudorandom number generator.
    gammas : list, optoinal
        List of step sizes of the gradient decent.
        Default gamma = None.
    max_iters : list, optional
        List of maximum numbers of iteration.
        Default max_iters = None.
    lambda_ : list, optional
        List of regularization parameters.
        Default lambda_ = None.
    degree : list, optional
        List of polynomial degrees of the features.
        If degree is equal to 0, polynomical transformation isn't applied to the features.
        Default degree = None.
    
    Returns:
    --------
    Plot describing the accuracy for different parameters
    """
    np.random.seed(seed)
    
    k_fold = 3
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    global_acc = []
    
    if type_ in ['GD', 'SGD', 'LR']:
        params_1 = max_iters
        params_2 = gammas
        label_ = 'max_iters = '
        xlabel = 'gamma'
    elif type_ in ['RR']:
        params_1 = degrees
        params_2 = lambdas
        label_ = 'degree = '
        xlabel = 'lambda'
    elif type_ in ['RLR']:
        params_1 = degrees
        params_2 = lambdas
        label_ = 'degree = '
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
                    raise TypeError(f"{type_} Wrong type!")
                acc_batch.append(acc_)
            acc.append(np.mean(acc_batch))
        global_acc.append(acc)

    print("Accuracy is {:.4f}".format(np.max(global_acc)))
    
    #plotting results
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(len(params_1)):
        label = label_ + str(params_1[i])
        if xlabel == 'gamma' and type_ != 'LR':
            plt.plot(params_2, global_acc[i], marker=".", label=label)
        else:
            plt.semilogx(params_2, global_acc[i], marker=".", label=label)
            plt.xlim(10**bd_left, 10**bd_right)
    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel("Accuracy", fontsize=17)
    plt.title("Cross validation for the " + type_, fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig('pictures/cross_val_basline'+type_+'.png')