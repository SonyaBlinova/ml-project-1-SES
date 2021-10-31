import numpy as np
from utils import *


def preproccess(input_data, labels):
    """
    Split dataset into 4 sets, deleting non-informative columns and replacing -999 with mean values.
    
    Parameters:
    -----------
    input_data : ndarray
        Matrix of features.
    labels : ndarray
        Target values belonging to the set {-1, 1}.
    
    Returns:
    --------
    jet_groups : list
        List of matrices of features after preprocess.
    jet_labels : list
        List of corresponding labels for each group of features.
    del_columns : dict
        Columns that should be deleted in the test.
    mean_of_all_col : list
        Mean values for each column for each group.
    """
    jets = np.unique(input_data.T[22])
    jet_groups = []
    jet_labels = []
    del_columns = {}
    mean_of_all_col = [[], [], [], []]
    data_labels = np.concatenate((input_data, labels.reshape(-1, 1)), axis=1)
    for jet in jets:
        del_columns[str(int(jet))] = []
        jet_group_ = data_labels[data_labels[:, 22] == jet]
        jet_group, jet_label = jet_group_[:, :input_data.shape[1]], jet_group_[:, -1]
        for i in range(jet_group.shape[1]):
            if len(np.unique(jet_group[:, i])) == 1:
                del_columns[str(int(jet))].append(i)
            else:
                jet_group[:, i][jet_group[:, i] == -999] = np.nan
                jet_group[:, i] = np.where(np.isnan(jet_group[:, i]), np.nanmean(jet_group[:, i]), jet_group[:, i])
                mean_of_all_col[int(jet)].append(np.nanmean(jet_group[:, i]))
        for col in del_columns[str(int(jet))][::-1]:
            jet_group = np.delete(jet_group, col, 1)
        
        jet_groups.append(jet_group)
        jet_labels.append(jet_label)
    print("these are the deleted columns: ",del_columns) 
    return jet_groups, jet_labels, del_columns, mean_of_all_col

def correletions(jet_groups_cor):
    """
    Find correlated columns and delete them.
    
    Parameters:
    -----------
    jet_groups_cor : list
        List of matrices of features for each group.
        
    Returns:
    --------
    jet_groups_cor : list
        List of matrices of features for each group without correlated columns.
    del_columns_cor : list
        Columns that should be deleted in the test for each group.
    """
    correlations = [[], [], [], []]
    del_columns_cor = {}
    for jet in range(4):
        del_columns_cor[str(int(jet))] = []
        for i in range(jet_groups_cor[jet].shape[1]):
            for j in range(i+1, jet_groups_cor[jet].shape[1]):
                corr = np.corrcoef(jet_groups_cor[jet][:, i], jet_groups_cor[jet][:, j])[0][1]
                if corr > 0.8:
                    correlations[jet].append(j)
        correlations[jet] = np.sort(np.unique(correlations[jet]))
        for col in correlations[jet][::-1]:
            del_columns_cor[str(int(jet))].append(col)
            jet_groups_cor[jet] = np.delete(jet_groups_cor[jet], col, 1)
    print("these are the deleted columns: ",del_columns_cor) 
    return jet_groups_cor, del_columns_cor

def normalization(jet_groups, jet_labels):
    """
    Data normalization.
    
    Parameters:
    -----------
    jet_groups : list
        List of matrices of features for each group.
    jet_labels : list
        List of corresponding labels for each group of features.
        
    Returns:
    --------
    jet_groups : list
        List of matrices of features for each group after normalization.
    jet_labels : list
        List of corresponding labels for each group of features.
    means : list
        List of means for each group for test normalization.
    stds : list
        List of stds for each group for test normalization.
    """
    means = []
    stds = []
    for i, (jet_group, jet_label) in enumerate(zip(jet_groups, jet_labels)):
        jet_group, mean_x, std_x = standardize(jet_group)
        means.append(mean_x)
        stds.append(std_x)
        jet_label, jet_group = build_model_data(jet_group, jet_label)
        jet_groups[i] = jet_group
        jet_labels[i] = jet_label
        print(f'jet : {i+1}, shape y : {jet_label.shape}, shape x : {jet_group.shape}')
    return jet_groups, jet_labels, means, stds

def remove_outliers(jet_groups,jet_label):
    """
    Removing outliers.
    
    Parameters:
    -----------
    jet_groups : list
        List of matrices of features for each group.
    jet_labels : list
        List of corresponding labels for each group of features.
        
    Returns:
    --------
    jet_groups : list
        List of matrices of features for each group without outliers.
    jet_labels : list
        List of corresponding labels for each group of features without outliers.
    """
    l1 = {0: set(), 1: set(), 2: set(), 3: set()}
    for i in range(4):
        for j in range(jet_groups[i].shape[1]):
            mean = np.mean((jet_groups[i])[:,j])
            standard_deviation = np.std((jet_groups[i])[:,j])
            max_deviations = 5
            for k in range(jet_groups[i].shape[0]):
                if(np.abs((jet_groups[i])[k,j] - mean) > max_deviations * standard_deviation ) :
                    l1[i].add(k) 

        jet_groups[i]  = np.delete(jet_groups[i], list(l1[i]), 0)
        jet_label[i]  = np.delete(jet_label[i], list(l1[i]), 0)      
    return jet_groups, jet_label

def preproccess_test(input_data, ids_test, del_columns, del_columns_cor, means, stds, mean_of_all_col):
    """
    Preprocess test data.
    
    Parameters:
    -----------
    input_data : ndarray
        Matrix of features.
    ids_test : ndarray
        Indeces for the test data.
    del_columns : list
        List of columns should be remove after splitting test features into 4 groups.
    del_columns_cor : list
        List of columns should be removed in the test features.
    means : list
        List of the means for normalization.
    stds : list
        List of the stds for normalization.
    mean_of_all_col : list
        List of means for each column for each group. Used for -999 removal.
        
    Returns:
    --------
    jet_groups : list
        List of matrices of features for each group after preprocess.
    jet_idxs : list
        List of corresponding indeces for each group of features after preprocess.
    """
    jets = np.unique(input_data.T[22])
    jet_groups = []
    jet_idxs = []
    data_labels = np.concatenate((input_data, ids_test.reshape(-1, 1)), axis=1)
    for i in range(4):
        jet_group_ = data_labels[data_labels[:, 22] == i]
        jet_group, jet_idx = jet_group_[:, :input_data.shape[1]], jet_group_[:, -1]
        for col in del_columns[str(i)][::-1]:
            jet_group = np.delete(jet_group, col, 1)
        for col in range(jet_group.shape[1]):
            jet_group[:, col][jet_group[:, col] == -999] = np.nan
            jet_group[:, col] = np.where(np.isnan(jet_group[:, col]), mean_of_all_col[i][col], jet_group[:, col])
        for col in del_columns_cor[str(i)]:
            jet_group = np.delete(jet_group, col, 1)
        
        jet_group = (jet_group - means[i])/stds[i]
        jet_group = np.concatenate((np.ones(jet_group.shape[0]).reshape(-1, 1), jet_group), axis=1)
        
        jet_groups.append(jet_group)
        jet_idxs.append(jet_idx)
    jet_idxs = np.array(np.concatenate(jet_idxs))
    return jet_groups, jet_idxs
