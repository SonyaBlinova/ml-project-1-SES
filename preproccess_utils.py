import numpy as np
from utils import *


def preproccess(input_data, labels):
    jets = np.unique(input_data.T[22])
    jet_groups = []
    jet_labels = []
    del_columns = {}
<<<<<<< HEAD
=======
    mean_of_all_col = [[], [], [], []]
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195
    data_labels = np.concatenate((input_data, labels.reshape(-1, 1)), axis=1)
    for jet in jets:
        del_columns[str(int(jet))] = []
        jet_group_ = data_labels[data_labels[:, 22] == jet]
        jet_group, jet_label = jet_group_[:, :input_data.shape[1]], jet_group_[:, -1]
<<<<<<< HEAD
        for i in range(len(jet_group[0])):
=======
        for i in range(jet_group.shape[1]):
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195
            if len(np.unique(jet_group[:, i])) == 1:
                del_columns[str(int(jet))].append(i)
            else:
                jet_group[:, i][jet_group[:, i] == -999] = np.nan
                jet_group[:, i] = np.where(np.isnan(jet_group[:, i]), np.nanmean(jet_group[:, i]), jet_group[:, i])
<<<<<<< HEAD
=======
                mean_of_all_col[int(jet)].append(np.nanmean(jet_group[:, i]))
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195
        for col in del_columns[str(int(jet))][::-1]:
            jet_group = np.delete(jet_group, col, 1)
        
        jet_groups.append(jet_group)
        jet_labels.append(jet_label)
<<<<<<< HEAD
    return jet_groups, jet_labels, del_columns
=======
    return jet_groups, jet_labels, del_columns, mean_of_all_col
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195

def correletions(jet_groups_cor):
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
    return jet_groups_cor, del_columns_cor

def normalization(jet_groups, jet_labels):
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

<<<<<<< HEAD
def preproccess_test(input_data, ids_test, del_columns, del_columns_cor, means, stds, means_for_outliers, stds_for_outliers):
    jets = np.unique(input_data.T[22])
    jet_groups = []
    jet_idxs = []
    l1 = {0: set(), 1: set(), 2: set(), 3: set()}
=======
def preproccess_test(input_data, ids_test, del_columns, del_columns_cor, means, stds, mean_of_all_col):
    jets = np.unique(input_data.T[22])
    jet_groups = []
    jet_idxs = []
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195
    data_labels = np.concatenate((input_data, ids_test.reshape(-1, 1)), axis=1)
    for i in range(4):
        jet_group_ = data_labels[data_labels[:, 22] == i]
        jet_group, jet_idx = jet_group_[:, :input_data.shape[1]], jet_group_[:, -1]
        for col in del_columns[str(i)][::-1]:
            jet_group = np.delete(jet_group, col, 1)
<<<<<<< HEAD
        for col in del_columns_cor[str(i)]:
            jet_group = np.delete(jet_group, col, 1)
        
        for j in range(jet_group.shape[1]):
            max_deviations = 5
            for k in range(jet_group.shape[0]):
                if(np.abs(jet_group[k,j] - means_for_outliers[i]) > max_deviations * stds_for_outliers[i] ):
                    l1[i].add(k) 
        jet_group  = np.delete(jet_group, list(l1[i]), 0)
        jet_idx  = np.delete(jet_idx, list(l1[i]), 0)
        
=======
        for col in range(jet_group.shape[1]):
            jet_group[:, col][jet_group[:, col] == -999] = np.nan
            jet_group[:, col] = np.where(np.isnan(jet_group[:, col]), mean_of_all_col[i][col], jet_group[:, col])
        for col in del_columns_cor[str(i)]:
            jet_group = np.delete(jet_group, col, 1)
        
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195
        jet_group = (jet_group - means[i])/stds[i]
        jet_group = np.concatenate((np.ones(jet_group.shape[0]).reshape(-1, 1), jet_group), axis=1)
        
        jet_groups.append(jet_group)
        jet_idxs.append(jet_idx)
    jet_idxs = np.array(np.concatenate(jet_idxs))
    return jet_groups, jet_idxs

#remove outliers  : 
def remove_outliers(jet_groups,jet_label):
    l1 = {0: set(), 1: set(), 2: set(), 3: set()}
<<<<<<< HEAD
    means = []
    stds = []
    for i in range(4):
        print("shape befor jet ", i+1, jet_groups[i].shape[0])
=======
    for i in range(4):
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195
        for j in range(jet_groups[i].shape[1]):
            mean = np.mean((jet_groups[i])[:,j])
            standard_deviation = np.std((jet_groups[i])[:,j])
            max_deviations = 5
            for k in range(jet_groups[i].shape[0]):
                if(np.abs((jet_groups[i])[k,j] - mean) > max_deviations * standard_deviation ) :
                    l1[i].add(k) 
<<<<<<< HEAD
            means.append(mean)
            stds.append(standard_deviation)
        jet_groups[i]  = np.delete(jet_groups[i], list(l1[i]), 0)
        jet_label[i]  = np.delete(jet_label[i], list(l1[i]), 0)
        print("shape after", jet_groups[i].shape[0])
        
    return jet_groups, jet_label, means, stds
=======
        jet_groups[i]  = np.delete(jet_groups[i], list(l1[i]), 0)
        jet_label[i]  = np.delete(jet_label[i], list(l1[i]), 0)      
    return jet_groups, jet_label
>>>>>>> 5f36c634eb2b4dc44cb29474d8a24be3e3361195
