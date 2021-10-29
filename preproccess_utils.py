import numpy as np
from utils import *


def preproccess(input_data, labels):
    jets = np.unique(input_data.T[22])
    jet_groups = []
    jet_labels = []
    del_columns = {}
    data_labels = np.concatenate((input_data, labels.reshape(-1, 1)), axis=1)
    for jet in jets:
        del_columns[str(int(jet))] = []
        jet_group_ = data_labels[data_labels[:, 22] == jet]
        jet_group, jet_label = jet_group_[:, :input_data.shape[1]], jet_group_[:, -1]
        for i in range(len(jet_group[0])):
            if len(np.unique(jet_group[:, i])) == 1:
                del_columns[str(int(jet))].append(i)
            else:
                jet_group[:, i][jet_group[:, i] == -999] = np.nan
                jet_group[:, i] = np.where(np.isnan(jet_group[:, i]), np.nanmean(jet_group[:, i]), jet_group[:, i])
        for col in del_columns[str(int(jet))][::-1]:
            jet_group = np.delete(jet_group, col, 1)
        
        jet_groups.append(jet_group)
        jet_labels.append(jet_label)
    return jet_groups, jet_labels, del_columns

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

#remove outliers  : 
def remove_outliers(jet_groups,jet_label):
    outliers=[]
    l=[]
    print("shape befor", jet_groups[0].shape[0])
    l1={0:set(),1:set(),2:set(),3:set()}
    for i in range(4):
        outlier=[]
        for j in range(jet_groups[i].shape[1]):
            mean = np.mean((jet_groups[i])[:,j])
            standard_deviation = np.std((jet_groups[i])[:,j])
            distance_from_mean = np.abs((jet_groups[i])[:,j] - mean)
            max_deviations = 5
            for k in range(jet_groups[i].shape[0]):
                if( ((jet_groups[i])[k,j]-mean)> max_deviations * standard_deviation ) :
                    l1[i].add(k) 
            
                    
      
           # outlier = distance_from_mean > max_deviations * standard_deviation
            #column = (jet_groups[i][:,j])[outlier]
    
        jet_groups[i]  = np.delete(jet_groups[i],list(l1[i]),0)
        jet_label[i]  = np.delete(jet_label[i],list(l1[i]),0)
        
        
    print("shape after", jet_groups[0].shape[0])
    return jet_groups,jet_label
            
        
        
     


