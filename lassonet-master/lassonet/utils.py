import matplotlib.pyplot as plt
from functools import partial

def linear_generator(p, N, s, K=1):
    directory = '/mnt/ufs18/home-033/gangulia/lassonet-master/lassonet/data/linear/my_version/p_' + str(p) + '_N_' + str(N) + '_s_' + str(s) 
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for k in range(K):
        rho = 0.7
        cov=torch.rand(p,p)
        for i in np.arange(p):
            for j in np.arange(p):
               cov[i,j]=pow(rho, abs((i+1)-(j+1)))
        
        X = np.random.multivariate_normal(np.zeros(p), cov, N)
        m = np.sqrt(2 * np.log(p)/N)
        M = 100 * m
        #e=np.random.multivariate_normal(mean, cov, 5000).T
        beta = np.zeros(p)
        #non_zero = np.random.choice(p, s, replace=False)
        non_zero = np.arange(s)*10
        beta[non_zero] = np.random.uniform(m, M, s)
        e = np.random.normal(0, np.sqrt(np.var(X.dot(beta))*(1/9)), N)
        y = X.dot(beta) + e
        X_all = X
        fn_X = directory + '/X_' + str(k) + '.txt'
        fn_y = directory + '/y_' + str(k) + '.txt'
        fn_beta = directory + '/beta_' + str(k) + '.txt'
        np.savetxt(fn_X, X_all)
        np.savetxt(fn_y, y)
        np.savetxt(fn_beta, beta)
 
def data_load_l(k, normalization=True, directory = "/mnt/ufs18/home-033/gangulia/lassonet-master/lassonet/data/linear/my_version/p_1000_N_800_s_10/"):
    # Directory for the datasets
    x = np.loadtxt(directory+'X_'+str(k)+'.txt')
    y = np.loadtxt(directory+'y_'+str(k)+'.txt')
    beta = np.loadtxt(directory+'beta_'+str(k)+'.txt')
    n = x.shape[0]
    # Take last 500 samples as testing set
    supp = np.where(beta != 0)[0]
    x_test = x[int(n/2):]
    y_test = y[int(n/2):]
    # Take first 500 samples as training set
    x = x[:int(n/2)]
    y = y[:int(n/2)]
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
