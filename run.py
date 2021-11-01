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
import argparse
sns.set()


parser = argparse.ArgumentParser()
parser.add_argument("-path", type=str, dest="data_path", help="Path to folder", required=True)
args = parser.parse_args()
print(args)
# loading data

DATA_TRAIN_PATH = args.data_path#'../data/train.csv'
labels, input_data, ids = load_csv_data(DATA_TRAIN_PATH)
seed = 46
ratio = 0.8
x_train, y_train, x_test, y_test = split_data(input_data, labels, ratio=ratio, seed=seed)

# standardize train data : 
x_train, mean_x, std_x = standardize(x_train)
y_train, x_train = build_model_data(x_train, y_train)

# standardize test data :
x_test = (x_test - mean_x)/std_x
y_test, x_test = build_model_data(x_test, y_test)

print('------------------base models-------------------------------')
#---------------------Least squares GD-------------------------
max_iters = 1000
gamma = 0.07

np.random.seed(seed)
initial_w = np.random.rand((x_train.shape[1]))
w, loss = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma, plot_loss = False)
y_pred = predict_labels(w, x_test)

print("Least squares GD accuracy :", accuracy(y_pred, y_test))

#---------------------Least squares SGD-------------------------
max_iters = 1300
gamma = 0.5

np.random.seed(seed)
initial_w = np.random.rand((x_train.shape[1]))
w, loss = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma, plot_loss = False)
y_pred = predict_labels(w, x_test)

print("Least squares SGD accuracy : ", accuracy(y_pred, y_test))

#---------------------Least squares exact-------------------------
w, loss = least_squares(y_train, x_train)
y_pred = predict_labels(w, x_test)

print("Least squares exact accuracy :", accuracy(y_pred, y_test))

#---------------------Ridge regression-------------------------
lambda_ = 1e-5
w, loss = ridge_regression(y_train, x_train, lambda_)
y_pred = predict_labels(w, x_test)

print("Ridge regression accuracy :", accuracy(y_pred, y_test))

#---------------------Logistic Regression-------------------------
max_iters = 1000
gamma = 0.5

np.random.seed(seed)
initial_w = np.random.rand((x_train.shape[1]))
w, loss = logistic_regression(y_train, x_train, initial_w, max_iters, gamma, plot_loss = False)
y_pred = predict_labels_for_log(w, x_test)
              
print("Logistic Regression accuracy :", accuracy(y_pred, y_test))

#---------------------Logistic Regression with Regularization------------
max_iters = 1000
gamma = 0.5
lambda_ = 1e-6

np.random.seed(seed)
initial_w = np.random.rand((x_train.shape[1]))
w, loss = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma, plot_loss = False)
y_pred = predict_labels_for_log(w, x_test)
              
print("Logistic Regression with Regularization accuracy :", accuracy(y_pred, y_test))


































