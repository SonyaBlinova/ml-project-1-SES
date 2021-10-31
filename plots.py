import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.xlabel("lambdas")
    plt.ylabel("error")
    plt.legend(loc=1)
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")

def confusion_matrix(y_tests,y_preds):
    c=[[],[]]
    c[0].append(np.sum((y_tests==-1) & (y_preds==-1)))
    c[1].append(np.sum((y_tests==-1) & (y_preds==1)))
    c[0].append(np.sum((y_tests==1) & (y_preds==-1)))
    c[1].append(np.sum((y_tests==1) & (y_preds==1)))
  
    # plt.figure(figsize=(10,7))
    fig, ax = plt.subplots()
    expected = ["-1","1"]
    predicted = ["-1","1"]
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(c,annot=True, annot_kws={"size": 10}) # font size
    plt.xlabel("Expected labels \n \n The resulted accuracy : "+str(accuracy(y_preds, y_tests)*100)+"%  , The resulted F1 score: "+str(F1_score(y_preds, y_tests)*100)+"%")  
    plt.ylabel("Prediceted labels")
    ax.set_xticklabels(expected)
    ax.set_yticklabels(predicted)
    ax.set_title('Confusion matrix')
    plt.show()
    
def correlation_plot(data):
    R=np.zeros((data.shape[1],data.shape[1]))
    for i in range(0,data.shape[1]):
        for j in range(0,data.shape[1]):
            if(i<=j):
                R[i,j]=np.corrcoef(data[:,i],data[:,j])[0,1]
                
    # plt.figure(figsize=(10,7))
    fig, ax = plt.subplots()
    sns.set(font_scale=1) # for label size
    sns.heatmap(R,annot_kws={"size": 10}) # font size
    plt.xlabel("features")
    plt.ylabel("features")
    ax.set_title('Correlation matrix')
    plt.show()
    
