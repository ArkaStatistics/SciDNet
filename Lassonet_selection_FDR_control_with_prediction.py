#!/usr/bin/env python
import sys
import os
sys.path.append("/mnt/ufs18/home-033/gangulia/lassonet-master")
import argparse
from time import clock
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils_simulation import data_generator, data_load_new,for_pred
from lassonet import LassoNetRegressor, plot_path
from random import choices
from joblib import Parallel, delayed
import itertools
from itertools import chain
from numpy.random import seed
import tensorflow
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2
import time
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


print("a")
def get_args():
    '''
    This function returns the arguments from terminal and set them to display
    '''
    parser = argparse.ArgumentParser(
        description = 'Run Lassonet for FDR control',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )     
    parser.add_argument('--jobID', default = 1,
        help = 'enter jobID')
    return parser.parse_args()

args=get_args()



dirc = "/mnt/ufs18/home-033/gangulia/Work/Rep_" + str(args.jobID) + "/"
X, Y, X_all, Y_all, supp_true, grp_length, g_unlist, active_set = data_load_new(directory=dirc)
_, true_features = X.shape
feature_names=np.arange(true_features)
n,p = X.shape
n_features=p
#X_train, Y_train, X_test, Y_test=X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy()


X = StandardScaler().fit_transform(X)
y = scale(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, y)

start_time = time.time()
#group formation
g=list(np.arange(len(active_set)))
k=0
for i in range(len(active_set)):
    g[i]= g_unlist[k:(k+grp_length[i])]
    k= k+grp_length[i]

active_index=list(itertools.chain(*g))
#generating bootstrap datasets
index = np.random.choice(np.arange(n),n)


def importance_bootstrap(iter):
    index = np.random.choice(np.arange(n),n)
    model = LassoNetRegressor(
        hidden_dims=(10,),
        eps_start=0.01,
        verbose=True,
    )
    path = model.path(X[index], Y[index])
    importances = model.feature_importances_.numpy()
    order = np.argsort(importances)[::-1]
    importance=[]
    for i in range(n_features):
        importance.append(list(order).index(i))
    
    return importance


B=50
results = Parallel(n_jobs=10)(delayed(importance_bootstrap)(i) for i in range(B))

def get_importance(i):
    importance_i=[]
    for j in range(B):
        importance_i.append(results[j][i])
     
    return np.mean(importance_i), np.var(importance_i)

importance=[]
importance_var=[]
for i in range(n_features):
    importance.append(get_importance(i)[0])
    importance_var.append(get_importance(i)[1])


def sorted_enumerate_pos(seq):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]


def sorted_enumerate_val(seq):
    return [v for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

imp_pos= sorted_enumerate_pos(importance)
ordered_imp= sorted_enumerate_val(importance)
ordered_var= sorted_enumerate_val(importance_var)


results_ordered=np.zeros((B,n_features))
for b in range(B):
    results_ordered[b]=list( np.array(results[b], dtype=object)[imp_pos] )
   
        

def fdr(i):
    sel_pos=imp_pos[:i]
    count= []
    for index in range(i):
        c=[]
        for b in range(B):
            if (results_ordered[b][index]<(ordered_imp[index]- 3) or results_ordered[b][index]>(ordered_imp[index]+3)): c.append(1)
            else: c.append(0)
        
        count.append(np.sum(c)/B)
    
    return 2*np.sum(count)/len(sel_pos)
        


FDR=[]
for pos in (np.arange(p)+1):
    FDR.append(fdr(pos))   

# Record the end time
end_time = time.time()

# Calculate the runtime
runtime = end_time - start_time
sel_pos=  active_set[np.array(imp_pos)[np.where(np.array(FDR)<.2)]] 
active_group= []
for i in range(len(sel_pos)):
    for j in range(len(active_set)):
        if (sel_pos[i] in set(g[j])): active_group.append(g[j])

power=[]
fpr=[]
threshold= [.1,.2]
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

n_var=[]
n_clust=[]
for t in threshold:
    if(np.min(FDR)<=t):
        sel_pos=  active_set[np.array(imp_pos)[np.where(np.array(FDR)<t)]]    
        active_group= []
        for i in range(len(sel_pos)):
            for j in range(len(active_set)):
                if (sel_pos[i] in set(g[j])): active_group.append(g[j])
        
        
        all_pos=list(itertools.chain(*active_group))
        n_var.append(len(all_pos))
        n_clust.append(len(active_group))
        if(len(sel_pos)>0): power.append(len(intersection(all_pos, supp_true))/len(supp_true))
        else: power.append(0)
        fd=[]
        for i in range(len(active_group)):
            if(len(intersection(active_group[i], supp_true))>0): fd.append(0)
            else: fd.append(1)
        
        if(len(sel_pos)>0): fpr.append(sum(fd)/len(sel_pos))
        else: fpr.append(0)
    
    else:
        power.append(0)
        fpr.append(0)
        n_var.append(0)
        n_clust.append(0)

SciDNet_stat= [power, fpr, n_var, n_clust]
########################################################################################
#LassoNet checking
# standardize

X = StandardScaler().fit_transform(X_all)
y = scale(Y_all)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y_all)

model = LassoNetRegressor(hidden_dims=(10,), verbose=True)
path = model.path(X_train[:,(active_set-1)], Y_train)
n_selected = []
mse = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test[:,(active_set-1)])
    n_selected.append(save.selected.sum().cpu().numpy())
    mse.append(mean_squared_error(Y_test, y_pred))
    lambda_.append(save.lambda_)

best_iter= np.argsort(mse)[0]
sel_pos_LassoNet_screen= active_set[np.where(path[best_iter].selected)[0]]
pred_screening_LN= mse[best_iter]
active_group_LN= []
for i in range(len(sel_pos_LassoNet_screen)):
    for j in range(len(active_set)):
        if (sel_pos_LassoNet_screen[i] in set(g[j])): active_group_LN.append(g[j])


all_pos_LN=list(itertools.chain(*active_group_LN))
if(len(sel_pos_LassoNet_screen)>0): power_LN_screen=(len(intersection(all_pos_LN, supp_true))/len(supp_true))
else: power_LN_screen=0

fd=[]
for i in range(len(active_group_LN)):
    if(len(intersection(active_group_LN[i], supp_true))>0): fd.append(0)
    else: fd.append(1)

if(len(sel_pos_LassoNet_screen)>0): fpr_LN_screen=(sum(fd)/len(sel_pos_LassoNet_screen))
else: fpr_LN_screen=0

size_screening_LN= n_selected[best_iter]
LassoNet_screening_stat=[power_LN_screen, fpr_LN_screen, size_screening_LN.tolist(),len(all_pos_LN)]



directory = '/mnt/ufs18/home-033/gangulia/Work/Rep_' + str(args.jobID) 
if not os.path.exists(directory):
    os.makedirs(directory)
  

#np.savetxt(directory + '/Output_power.csv', power, delimiter=",")
#np.savetxt(directory + '/Output_fpr.csv', fpr, delimiter=",")
np.savetxt(directory + '/Output_SciDNet_stat.csv', SciDNet_stat, fmt='%s')
np.savetxt(directory + '/Output_LassoNet_screening_stat.csv', LassoNet_screening_stat, delimiter=",")
#np.savetxt(directory + '/Output_MLP_pred_error.csv', pred_error_MLP, delimiter=",")
