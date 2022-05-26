import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import pandas as pd
import tensorflow as tf


import matplotlib.pyplot as plt
from functools import partial

def data_generator(p, N, s, K=1):
    directory = '...' 
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for k in range(K):
        rho = 0.8
        cov=torch.rand(p,p)
        for i in np.arange(p):
            for j in np.arange(p):
               cov[i,j]=pow(rho, abs((i+1)-(j+1)))
        
        X = np.random.multivariate_normal(np.zeros(p), cov, N)
        m = 2
        M = 4
        #e=np.random.multivariate_normal(mean, cov, 5000).T
        beta = np.zeros(p)
        #non_zero = np.random.choice(p, s, replace=False)
        non_zero = np.arange(s)*10
        beta[non_zero] = np.random.uniform(m, M, s)
        e = np.random.normal(0, np.sqrt(np.var(np.exp(X[:, 0]) + 10*np.absolute(X[:, 10]) + 5*np.sin(X[:, 20]) + 4*X[:, 30] + X[:, 40]**3)*(7/3)), N)
        #e = np.random.normal(0, 1, N)
        #y = np.exp(X.dot(beta))/(1+np.exp(X.dot(beta))) + e
        #y=np.exp(X[:, 0]) + (X[:, 1])**3 + 5*np.sin(X[:, 2]) + 4*(X[:, 3] * X[:, 4]) + e
        y=np.exp(X[:, 0]) + 10*np.absolute(X[:, 10]) + 5*np.sin(X[:, 20]) + 4*X[:, 30] + X[:, 40]**3 + e
        #y = X.dot(beta) + e
        X_all = X
        fn_X = directory + '/X_' + str(k) + '.txt'
        fn_y = directory + '/y_' + str(k) + '.txt'
        fn_beta = directory + '/beta_' + str(k) + '.txt'
        np.savetxt(fn_X, X_all)
        np.savetxt(fn_y, y)
        np.savetxt(fn_beta, beta)
 
def data_load(k, normalization=True, directory = "/mnt/ufs18/home-033/gangulia/lassonet-master/lassonet/data/linear/my_version/p_1000_N_800_s_10/"):
    # Directory for the datasets
    x = np.loadtxt(directory+'X_'+str(k)+'.txt')
    y = np.loadtxt(directory+'y_'+str(k)+'.txt')
    beta = np.loadtxt(directory+'beta_'+str(k)+'.txt')
    n = x.shape[0]
    # Take last 500 samples as testing set
    supp = np.where(beta != 0)[0]
    x_test = x[int(4*n/5):]
    y_test = y[int(4*n/5):]
    # Take first 500 samples as training set
    x = x[:int(4*n/5)]
    y = y[:int(4*n/5)]
    N, p = x.shape
    # normalize if needed
    if normalization:
        for j in range(p):
            x_test[:, j] = x_test[:, j]/np.sqrt(np.sum(x[:, j]**2)/float(N))
            x[:, j] = x[:, j]/np.sqrt(np.sum(x[:, j]**2)/float(N))
    X, Y = torch.Tensor(x), torch.Tensor(y)
    X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)
    Y = Y.view(-1, 1)
    X, Y = Variable(X), Variable(Y)
    X_test, Y_test = torch.Tensor(x_test), torch.Tensor(y_test)
    X_test, Y_test = X_test.type(torch.FloatTensor), Y_test.type(torch.FloatTensor)
    Y_test = Y_test.view(-1, 1)
    X_test, Y_test = Variable(X_test), Variable(Y_test)
    return X, Y, X_test, Y_test, supp


def data_load_new(normalization=True, directory = "/mnt/ufs18/home-033/gangulia/lassonet-master/lassonet/data/linear/my_version/p_1000_N_800_s_10/"):
    # extracting supp_true
    all_1= pd.read_csv(directory+"supp_true.csv")
    all_1.columns=['num','pos']
    supp_true=np.array(all_1.pos)
    # extracting grp_length
    all_2= pd.read_csv(directory+"grp_length.csv")
    all_2.columns=['num','len']
    grp_length=np.array(all_2.len)
    # extracting g_unlist
    all_3= pd.read_csv(directory+"g_unlist.csv")
    all_3.columns=['num','pos']
    g_unlist=np.array(all_3.pos)
    # extracting active_set
    all_4= pd.read_csv(directory+"active_set.csv")
    all_4.columns=['num','pos']
    active_set=np.array(all_4.pos)
    # extracting active_data
    all_5= pd.read_csv(directory+"active_data.csv")
    del all_5[all_5.columns[0]]
    y=np.array(all_5[all_5.columns[0]])
    x=np.array(all_5[all_5.columns[1:]])
    #extracting all data
    all= pd.read_csv(directory+"all.csv")
    del all[all.columns[0]]
    y_all=np.array(all[all.columns[0]])
    x_all=np.array(all[all.columns[1:]])    
    n,p = x_all.shape
    # Take first 4/5th samples as training set
    x_train = x_all[:int(4*n/5)]
    y_train = y_all[:int(4*n/5)]
    # Take last 1/5th samples as testing set
    x_test = x_all[int(4*n/5):]
    y_test = y_all[int(4*n/5):]
    
    X, Y = torch.Tensor(x), torch.Tensor(y)
    X, Y = X.type(torch.FloatTensor), Y.type(torch.FloatTensor)
    Y = Y.view(-1, 1)
    X, Y = Variable(X), Variable(Y)
    X_test, Y_test = torch.Tensor(x_test), torch.Tensor(y_test)
    X_test, Y_test = X_test.type(torch.FloatTensor), Y_test.type(torch.FloatTensor)
    Y_test = Y_test.view(-1, 1)
    X_test, Y_test = Variable(X_test), Variable(Y_test)
    X_train, Y_train = torch.Tensor(x_train), torch.Tensor(y_train)
    X_train, Y_train = X_train.type(torch.FloatTensor), Y_train.type(torch.FloatTensor)
    Y_train = Y_train.view(-1, 1)
    X_train, Y_train = Variable(X_train), Variable(Y_train)
    

    return X, Y, x_all, y_all, supp_true, grp_length, g_unlist, active_set



def plot_path(model, path, X_test, y_test, *, score_function=None):
    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score


    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    path
        output of model.path
    X_test : array-like
    y_test : array-like
    score_function : function or None
        if None, use score_function=model.score
        score_function must take as input X_test, y_test
    """
    if score_function is None:
        score_fun = model.score
    else:
        assert callable(score_function)

        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict(X_test))

    n_selected = []
    score = []
    lambda_ = []
    for save in path:
        model.load(save.state_dict)
        n_selected.append(save.selected.sum())
        score.append(score_fun(X_test, y_test))
        lambda_.append(save.lambda_)

    plt.figure(figsize=(8, 8))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, score, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("score")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, score, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.tight_layout()
    

def for_pred(seed,train_X,train_Y,val_X,val_Y,test_X,test_Y, 
       n_classes,n_hidden1,n_hidden2, 
       learning_rate,epochs,batch_size,num_batches,dropout,alpha):
    
    # get the number of variables
    n_dim=train_X.shape[1]
    
    # create placeholders, initialize weights and biases, and define the loss function
    X=tf.placeholder(tf.float64,[None,n_dim])
    Y=tf.placeholder(tf.float64,[None,n_classes])
    keep_prob=tf.placeholder(tf.float64)
    
    W1=0.01*np.random.randn(n_dim,n_hidden1).astype(np.float64)
    B1=np.random.randn(n_hidden1).astype(np.float64)
    W2=0.01*np.random.randn(n_hidden1,n_hidden2).astype(np.float64)
    B2=np.random.randn(n_hidden2).astype(np.float64)
    W3=0.01*np.random.randn(n_hidden2,n_classes).astype(np.float64)
    B3=np.random.randn(n_classes).astype(np.float64)
    
    w1=tf.Variable(W1,name="weights1")
    b1=tf.Variable(B1,name="biases1")
    w2=tf.Variable(W2,name="weights2")
    b2=tf.Variable(B2,name="biases2")
    w3=tf.Variable(W3,name="weights3")
    b3=tf.Variable(B3,name="biases3")
    
    out1=tf.nn.relu(tf.matmul(X,w1)+b1)
    out1=tf.nn.dropout(out1,keep_prob,seed=seed)
    out2=tf.nn.relu(tf.matmul(out1,w2)+b2)
    out2=tf.nn.dropout(out2,keep_prob,seed=seed)
    out3=tf.matmul(out2,w3)+b3
    
    cost=tf.square(out3-Y)
    loss=tf.reduce_mean(cost)
    op_train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    init=tf.global_variables_initializer()
    loss_val_trace=[]
    accuracy_val_trace=[]
    
    # training using a GL_alpha stopping criterion
    with tf.Session() as sess:
        sess.run(init)
        
        i=1
        for j in range(math.ceil(num_batches)):
            sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
        loss_train=sess.run(loss,feed_dict={X:train_X,Y:train_Y,keep_prob:1.})
        loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
        pred_val=sess.run(out3,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
        #accuracy_val=np.mean(np.square(pred_val-val_Y))
        loss_val_trace.append(loss_val)
        #accuracy_val_trace.append(accuracy_val)
        
        while i<epochs and loss_val/min(loss_val_trace)<1+alpha:
            i+=1
            for j in range(math.ceil(num_batches)):
                sess.run(op_train,feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size],keep_prob:dropout})
            loss_val=sess.run(loss,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
            pred_val=sess.run(out3,feed_dict={X:val_X,Y:val_Y,keep_prob:1.})
            #accuracy_val=np.mean(np.square(pred_val-val_Y))
            loss_val_trace.append(loss_val)
            #accuracy_val_trace.append(accuracy_val)
            
        # get the initial loss and accuracy on the test set (with all original variables)
        loss_test=sess.run(loss,feed_dict={X:test_X,Y:test_Y,keep_prob:1.})
        pred_test=sess.run(out3,feed_dict={X:test_X,Y:test_Y,keep_prob:1.})
        #accuracy_test=np.mean(np.square(pred_test-test_Y))
        
        initial=[loss_test,pred_test]
        print('epochs:',i)
        #print('initial test loss:',loss_test,'initial test accuracy:',accuracy_test,'\n')
        #print('loss_val:', loss_val, 'loss_train', loss_train)

    # return the initial loss and accuracy on the test set
    return initial

