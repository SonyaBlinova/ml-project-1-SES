import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

def confusion_matrix(y_tests,y_preds):
    c=[[],[]]
    c[0].append(np.sum((y_tests==-1) & (y_preds==-1)))
    c[1].append(np.sum((y_tests==-1) & (y_preds==1)))
    c[0].append(np.sum((y_tests==1) & (y_preds==-1)))
    c[1].append(np.sum((y_tests==1) & (y_preds==1)))
  
    # plt.figure(figsize=(10,7))
    fig, ax = plt.subplots(figsize=(10,7))
    expected = ["-1","1"]
    predicted = ["-1","1"]
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(c, annot=True, annot_kws={"size": 18}) # font size
    plt.xlabel("Expected labels \n \n The resulted accuracy : "+str(accuracy(y_preds, y_tests)*100)+"%  , The resulted F1 score: "+str(F1_score(y_preds, y_tests)*100)+"%", fontsize=20)  
    plt.ylabel("Prediceted labels", fontsize=20)
    ax.set_xticklabels(expected, fontsize=20)
    ax.set_yticklabels(predicted, fontsize=20)
    ax.set_title('Confusion matrix', fontsize=20)
    plt.savefig('pictures/confusion_matrix.png')
    plt.show()
    
def correlation_plot(data):
    all_R = []
    for d in range(4):
        R = np.zeros((data[d].shape[1], data[d].shape[1]))
        for i in range(0,data[d].shape[1]):
            for j in range(0,data[d].shape[1]):
                if i <= j:
                    R[i,j] = np.corrcoef(data[d][:,i],data[d][:,j])[0,1]
        all_R.append(R)
                
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize = (24, 4))
    for i in range(4):
        sns.set(font_scale=1) # for label size
        sns.heatmap(all_R[i], ax=ax[i])#, annot_kws={"size": 10}) # font size
        ax[i].set_xlabel("features")
        ax[i].set_ylabel("features")
        ax[i].set_title('Correlation matrix group' + str(i+1))
    plt.savefig('pictures/correlation_plot.png')
    plt.show()
    
