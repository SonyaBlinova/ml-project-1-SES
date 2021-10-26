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


def build_model_data(x, y):
    """
    Form (y,tX) to get regression data in matrix form.
    
    Parameters
    ----------
    x : ndarray
        Matrix of features.
    y : ndarray
        Target values belonging to the set {-1, 1}.
        
    Returns
    -------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    x : ndarray
        Matrix of features.
    """
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def standardize(x):
    """
    Standardize the original data set.
    
    Parameters
    ----------
    x : ndarray
        Matrix of features.
        
    Returns
    -------
    x : ndarray
        Matrix of features after standartization.
    mean_x : float
        Mean value of x.
    std_x : float
        Standart deviation of x.
    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def compute_mse(y, tx, w):
    """
    Computation the loss by mse.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    w : ndarray
        Model weights.
        
    Returns
    -------
    mse : float
        MSE loss.
    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_logistic_loss(y, tx, w):
    """
    Computation logistic regression loss.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    w : ndarray
        Model weights.
        
    Returns
    -------
    loss : float
        Logistic regression loss.
    """
    h = sigmoid(tx, w.T)
    loss = - 1/y.shape[0]*np.sum((y == 1)*np.log(h) + (y == -1)*np.log(1 - h))
    return loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    tx : ndarray
        Matrix of features.
    batch_size : int
        Size of the batch.
    num_batches : int
        Amount of batches. Default is 1.
    shuffle : bool
        If True, shuffle indices before splitting, otherwise not. Defaults to True.
        
    Returns
    -------
    shuffled_y : ndarray
        Target values, included in batch.
    shuffled_x : ndarray
        Matrix of features, included in batch.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


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
    
    return 1/(1 + np.exp(-tx@w.T))

def split_data(x, y, ratio, seed=1):
    """
    Dataset split based on the split ratio.
    
    Parameters
    ----------
    x : ndarray
        Matrix of features.
    y : ndarray
        Target values belonging to the set {-1, 1}.
    ratio : float
        Split ratio.
    seed : int
        The number used to initialize a pseudorandom number generator.
        
    Returns
    -------
    x_train : ndarray
        Matrix of features for train.
    y_train : ndarray
        Target values belonging to the set {-1, 1} for train.
    x_test : ndarray
        Matrix of features for test.
    y_test : ndarray
        Target values belonging to the set {-1, 1} for test.
    """
    # set seed
    np.random.seed(seed)
    conc_data = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    np.random.shuffle(conc_data)
    x, y = conc_data[:, :x.shape[1]], conc_data[:, -1]
    x_train, x_test = np.split(x, [int(ratio*len(x))])
    y_train, y_test = np.split(y, [int(ratio*len(y))])
    return x_train, y_train, x_test, y_test


def accuracy(y_pred, y_test):
    """
    Prediction accuracy.
    
    Parameters
    ----------
    y_pred : ndarray
        Predicted labels.
    y_test : ndarray
        Target labels.
        
    Returns
    -------
    accuracy : float
        The percentage of difference between two numpy arrays.
    """
    diff = y_test - y_pred
    return len(diff[diff==0])/len(diff)

def build_k_indices(y, k_fold, seed):
    """
    Buildong k indices for k-fold.
    
    Parameters
    ----------
    y : ndarray
        Target values belonging to the set {-1, 1}.
    k_fold : int
        Amount of folds
    seed : int
        The number used to initialize a pseudorandom number generator.
        
    Returns
    -------
    k_indices : ndarray
        Array of k indices.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Parameters
    ----------
    x : ndarray
        Matrix of features.
    degree : int
        Polynomical degree.
        
    Returns
    -------
    x_poly : ndarray
        Polynomial basis functions for input data x, for j=0 up to j=degree.
    """
    if degree == 0:
        return x
    matr = []
    for x_i in x:
        matr.append([x_i**j for j in range(degree+1)])
    return np.reshape(matr, (x.shape[0], x.shape[1]*(degree+1)))

def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")

