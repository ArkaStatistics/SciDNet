#!/usr/bin/env python
import sys
import os
sys.path.append(".../lassonet-master")
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



dirc = "..."
X, Y, X_all, Y_all, supp_true, grp_length, g_unlist, active_set = data_load_new(directory=dirc)
_, true_features = X.shape
feature_names=np.arange(true_features)
n,p = X.shape
#X_train, Y_train, X_test, Y_test=X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy()


#group formation
g=list(np.arange(len(active_set)))
k=0
for i in range(len(active_set)):
    g[i]= g_unlist[k:(k+grp_length[i])]
    k= k+grp_length[i]


model = LassoNetRegressor(
    hidden_dims=(10,),
    eps_start=0.1,
    verbose=True,
)
path = model.path(X, Y)

n_features = X.shape[1]
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
importance_main=[]
for i in range(n_features):
    importance_main.append(list(order).index(i))


#generating bootstrap datasets
index = np.random.choice(np.arange(n),n)


def importance_bootstrap(iter):
    index = np.random.choice(np.arange(n),n)
    model = LassoNetRegressor(
        hidden_dims=(10,),
        eps_start=0.1,
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
results = Parallel(n_jobs=15)(delayed(importance_bootstrap)(i) for i in range(B))

def get_importance(i):
    importance_i=[]
    for j in range(B):
        importance_i.append(results[j][i])
     
    return np.mean(importance_i)

importance=[]
for i in range(n_features):
    importance.append(get_importance(i))


def sorted_enumerate_pos(seq):
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]


def sorted_enumerate_val(seq):
    return [v for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

imp_pos= sorted_enumerate_pos(importance)
ordered_imp= sorted_enumerate_val(importance)


results_ordered=np.zeros((B,n_features))
for b in range(B):
    results_ordered[b]=list( np.array(results[b], dtype=object)[imp_pos] )
   
        

def fdr(i):
    sel_pos=imp_pos[:i]
    count= []
    for index in range(i):
        c=[]
        for b in range(B):
            if (results_ordered[b][index]<(ordered_imp[index]-20) or results_ordered[b][index]>(ordered_imp[index]+20)): c.append(1)
            else: c.append(0)
        
        count.append(np.sum(c)/B)
    
    return 2*np.sum(count)/len(sel_pos)
        


FDR=[]
for pos in (np.arange(n_features)+1):
    FDR.append(fdr(pos))   

reduced_sel_pos = np.array(imp_pos)[np.where(np.array(FDR)<.15)]
sel_pos=  active_set[np.array(imp_pos)[np.where(np.array(FDR)<.15)]]
active_group= []
for i in range(len(sel_pos)):
    for j in range(len(active_set)):
        if (sel_pos[i] in set(g[j])): active_group.append(g[j])


def pred_bootstrap(iter):
    index = np.random.choice(np.arange(n),n)
    X_bt, Y_bt= X[index], Y[index]
    X_train_bt = X_bt[:int(4*n/5)]
    Y_train_bt = Y_bt[:int(4*n/5)]
    # Take last 1/5th samples as testing set
    X_test_bt = X_bt[int(4*n/5):]
    Y_test_bt = Y_bt[int(4*n/5):]
    
    Y_train_bt=np.reshape(Y_train_bt, (-1,1))
    Y_test_bt=np.reshape(Y_test_bt, (-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    print(scaler_x.fit(X_train_bt))
    Xtrain_scale_bt=scaler_x.transform(X_train_bt)
    print(scaler_x.fit(X_test_bt))
    Xtest_scale_bt=scaler_x.transform(X_test_bt)
    print(scaler_y.fit(Y_train_bt))
    Ytrain_scale_bt=scaler_y.transform(Y_train_bt)
    print(scaler_y.fit(Y_test_bt))
    Y_test_scale_bt=scaler_y.transform(Y_test_bt)
    
    
    model = Sequential()
    model.add(Dense(len(reduced_sel_pos), input_dim=len(reduced_sel_pos), kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
      
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    history=model.fit(Xtrain_scale_bt[:,reduced_sel_pos], Ytrain_scale_bt, epochs=30, batch_size=50, verbose=1, validation_split=0.2)
    predictions = model.predict(Xtest_scale_bt[:,reduced_sel_pos])
    test_error_bt=np.mean(np.square(predictions-Y_test_scale_bt))
    
    
    return test_error_bt


B=50
pred_results = Parallel(n_jobs=15)(delayed(pred_bootstrap)(i) for i in range(B))
after_error = [np.mean(pred_results) , np.var(pred_results)**.5]

#initial prediction
all= pd.read_csv("/mnt/ufs18/home-033/gangulia/Variable_selection_Lassonet/tractography_data/"+"all.csv")
del all[all.columns[0]]
Y_all=np.array(all[all.columns[0]])
X_all=np.array(all[all.columns[1:]])    

def pred_bootstrap(iter):
    index = np.random.choice(np.arange(n),n)
    X_bt, Y_bt= X_all[index], Y_all[index]
    X_train_bt = X_bt[:int(4*n/5)]
    Y_train_bt = Y_bt[:int(4*n/5)]
    # Take last 1/5th samples as testing set
    X_test_bt = X_bt[int(4*n/5):]
    Y_test_bt = Y_bt[int(4*n/5):]
    
    Y_train_bt=np.reshape(Y_train_bt, (-1,1))
    Y_test_bt=np.reshape(Y_test_bt, (-1,1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    print(scaler_x.fit(X_train_bt))
    Xtrain_scale_bt=scaler_x.transform(X_train_bt)
    print(scaler_x.fit(X_test_bt))
    Xtest_scale_bt=scaler_x.transform(X_test_bt)
    print(scaler_y.fit(Y_train_bt))
    Ytrain_scale_bt=scaler_y.transform(Y_train_bt)
    print(scaler_y.fit(Y_test_bt))
    Y_test_scale_bt=scaler_y.transform(Y_test_bt)
    
    
    model = Sequential()
    model.add(Dense(X_train_bt.shape[1], input_dim=X_train_bt.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
      
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    history=model.fit(Xtrain_scale_bt, Ytrain_scale_bt, epochs=30, batch_size=50, verbose=1, validation_split=0.2)
    predictions = model.predict(Xtest_scale_bt)
    test_error_bt=np.mean(np.square(predictions-Y_test_scale_bt))
    
    
    return test_error_bt


B=50
pred_results = Parallel(n_jobs=15)(delayed(pred_bootstrap)(i) for i in range(B))
initial_error = [np.mean(pred_results) , np.var(pred_results)**.5]
pred_error= [initial_error, after_error]
power=[]
fpr=[]
threshold= [.01,.05,.1,.15,.2]
for t in threshold:
    sel_pos=  active_set[np.array(imp_pos)[np.where(np.array(FDR)<t)]]
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))
    
    active_group= []
    for i in range(len(sel_pos)):
        for j in range(len(active_set)):
            if (sel_pos[i] in set(g[j])): active_group.append(g[j])
    
    
    all_pos=list(itertools.chain(*active_group))
    if(len(sel_pos)>0): power.append(len(intersection(all_pos, supp_true))/len(supp_true))
    else: power.append(0)
    fd=[]
    for i in range(len(active_group)):
        if(len(intersection(active_group[i], supp_true))>0): fd.append(0)
        else: fd.append(1)
    
    if(len(sel_pos)>0): fpr.append(sum(fd)/len(sel_pos))
    else: fpr.append(0)



directory = '...' 
if not os.path.exists(directory):
    os.makedirs(directory)
  

np.savetxt(directory + '/Output_power.csv', power, delimiter=",")
np.savetxt(directory + '/Output_fpr.csv', fpr, delimiter=",")
np.savetxt(directory + '/Output_pred_error.csv', pred_error, delimiter=",")

