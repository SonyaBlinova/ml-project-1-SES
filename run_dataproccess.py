import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from proj1_helpers import *
from implementations import *
from plots import *
from utils import *
from validation import *
from preproccess_utils import *
import seaborn as sns
import os
sns.set()

if not os.path.exists('pictures'):
    os.mkdir('pictures')
    
    
DATA_TRAIN_PATH = '../data/train.csv'
labels, input_data, ids = load_csv_data(DATA_TRAIN_PATH)

seed = 46
ratio = 0.8
x_train, y_train, x_test, y_test = split_data(input_data, labels, ratio=ratio, seed=seed)

x_trains, y_trains, del_columns, mean_of_all_col = preproccess(x_train, y_train)

x_trains, del_columns_cor = correletions(x_trains)
x_trains, y_trains = remove_outliers(x_trains, y_trains)
x_trains, y_trains, means, stds = normalization(x_trains, y_trains)

x_tests, y_tests = preproccess_test(x_test, y_test, del_columns, del_columns_cor, means, stds, mean_of_all_col)


print('\n')
print('---------------------DataPreprocessed models--------------------------')
#---------------------Least squares GD-------------------------
max_iters = 1000
gamma = 0.09
y_preds = []
for i in range(4):
    np.random.seed(seed)
    initial_w = np.random.rand((x_trains[i].shape[1]))
    w, loss = least_squares_GD(y_trains[i], x_trains[i], initial_w, max_iters, gamma, plot_loss = False)

    y_pred = predict_labels(w, x_tests[i])
    y_preds.append(y_pred)
    
print("Least squares GD accuracy :", accuracy(np.array(np.concatenate(y_preds)), y_tests))

#---------------------Least squares SGD-------------------------

max_iters = 1500
gamma = 0.2
y_preds = []
for i in range(4):
    np.random.seed(seed)
    initial_w = np.random.rand((x_trains[i].shape[1]))
    w, loss = least_squares_SGD(y_trains[i], x_trains[i], initial_w, max_iters, gamma, plot_loss = False)

    y_pred = predict_labels(w, x_tests[i])
    y_preds.append(y_pred)
              
print("Least squares SGD accuracy :", accuracy(np.array(np.concatenate(y_preds)), y_tests))

#---------------------Least squares exact-------------------------
y_preds = []
for i in range(4):
    w, loss = least_squares(y_trains[i], x_trains[i])

    y_pred = predict_labels(w, x_tests[i])
    y_preds.append(y_pred)

print("Least squares exact accuracy :", accuracy(np.array(np.concatenate(y_preds)), y_tests))

#---------------------Ridge regression-------------------------

lambda_ = 1e-5
y_preds = []
for i in range(4):
    w, loss = ridge_regression(y_trains[i], x_trains[i], lambda_)
    y_pred = predict_labels(w, x_tests[i])
    y_preds.append(y_pred)
print("Ridge regression accuracy  :", accuracy(np.array(np.concatenate(y_preds)), y_tests))

#---------------------Logistic Regression-------------------------
max_iters = 1000
gamma = 0.5
y_preds = []
for i in range(4):
    np.random.seed(seed)
    initial_w = np.random.rand((x_trains[i].shape[1]))
    w, loss = logistic_regression(y_trains[i], x_trains[i], initial_w, max_iters, gamma, plot_loss = False)
    y_pred = predict_labels_for_log(w, x_tests[i])
    y_preds.append(y_pred)

print("Logistic Regression accuracy :", accuracy(np.array(np.concatenate(y_preds)), y_tests))

#---------------------Logistic Regression with Regularization------------

max_iters = 1000
gamma = 0.5
lambda_ = 1e-4
y_preds = []
w_lrr = []
for i in range(4):
    np.random.seed(seed)
    initial_w = np.random.rand((x_trains[i].shape[1]))
    w, loss = reg_logistic_regression(y_trains[i], x_trains[i], lambda_, initial_w, max_iters, gamma, plot_loss = False)
    y_pred = predict_labels_for_log(w, x_tests[i])
    y_preds.append(y_pred)
    w_lrr.append(w)
              
print("Logistic Regression with Regularization accuracy :", accuracy(np.array(np.concatenate(y_preds)), y_tests))
