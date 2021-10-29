# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 01:51:12 2021

@author: saadc
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import numpy.ma as ma

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here

labels, input_data, ids = load_csv_data(DATA_TRAIN_PATH)

array=input_data.copy()
array[input_data==-999]=np.nan
array = np.where(np.isnan(input_data), ma.array(input_data, mask=np.isnan(input_data)).mean(axis=0), input_data)
