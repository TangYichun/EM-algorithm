import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.cluster import KMeans


def updateW(X, Mu, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * stats.multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return W


def updatePi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi


def logLH(X, Pi, Mu, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * stats.multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))


# Plot 95% confidence interval for 2D data
"""def plot_clusters(X, Mu, Var, clusterNum):
    colors = ['b', 'g', 'r']
    #n_clusters = len(Mu)
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    for i in range(clusterNum):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ciX = stats.norm.interval(0.95, loc=Mu[i][0], scale=Var[i][0]**0.5)
        ciY = stats.norm.interval(0.95, loc=Mu[i][1], scale=Var[i][1]**0.5)
        ellipse = Ellipse(Mu[i], ciX[1]-ciX[0], ciY[1]-ciY[0], **plot_args)
        ax.add_patch(ellipse)

    for i in range(n_cluster):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
        ciX2 = stats.norm.interval(0.95, loc=mu2[i][0], scale=var2[i][0]**0.5)
        ciY2 = stats.norm.interval(0.95, loc=mu2[i][1], scale=var2[i][1]**0.5)
        ellipse = Ellipse(Mu[i], ciX2[1]-ciX2[0], ciY2[1]-ciY2[0], **plot_args)
        ax.add_patch(ellipse) 

    plt.show()"""


def updateMu(X, W):
    n_clusters = W.shape[1]
    Mu = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=W[:, i])
    return Mu


def updateVar(X, Mu, W):
    n_clusters = W.shape[1]
    Var = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=W[:, i])
    return Var


def kmeansInit(data, clusterNum):
    kmeans = KMeans(n_clusters=clusterNum, random_state=0)
    label = kmeans.fit_predict(data)
    #print("through")
    Mu = []
    Var = []
    Pi = []

    #Mu, Var, Pi = ([] for i in range(clusterNum))
    for i in range(clusterNum):
        cluster = data[label == i]
        Mu.append(list(np.mean(cluster, axis=0)))
        Var.append(list(np.var(cluster, axis=0)))
        Pi.append(len(cluster)/len(data))

    return Mu, Var, Pi

def choose_k(data, max_cluster, N, dim, start_time):
    bic = []
    bic_value = 0.0
    best_cluster = 0
    for cluster in range(1, max_cluster+1):
        if (time.time()-start_time > 10.0):
            #print("break")
            break
        Mu, Var, Pi = kmeansInit(data, cluster)
        W = np.ones((data.shape[0], cluster)) / cluster
        lnl = []
        for _ in range(20):
            if (time.time()-start_time > 10.0):
                print("10 sec limit")
                break
            #print("break")
            lnl.append(logLH(data, Pi, Mu, Var))
            #print("throughbic")
            W = updateW(data, Mu, Var, Pi)
            Pi = updatePi(W)
            Mu = updateMu(data, W)
            Var = updateVar(data, Mu, W)

        bic.append(-2 * lnl[-1] * N + np.log(N)*(dim * cluster))
    bic_value = min(bic)
    best_cluster = bic.index(min(bic))+1

    #print(bic)
    return best_cluster




def main():
    #filename = 'sample EM data.csv'
    filename, initial_clusterNum = input("Enter the filename (without blank) and the number of cluster (seperate with blank): ").split()
    #print(filename)
    #print(initial_clusterNum)
    data = np.genfromtxt(filename, delimiter=',')
    N, dim = data.shape
    #initial_clusterNum = int(initial_clusterNum)
    max_cluster = 20
    start_time = time.time()
    clusterNum = choose_k(data, max_cluster, N, dim, start_time)
    print("The optimal cluster number: ", clusterNum)

    """filename = 'label data.csv'
    X = np.genfromtxt(filename, delimiter=',')
    n_clusters = 3
    mu2 = np.zeros((n_clusters, 2))
    var2 = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        mu2[i]=np.mean(X[:,0:2][X[:,2]==i+1],axis=0)
        var2[i]=np.mean((X[:,0:2][X[:,2]==i+1] - mu2[i]) ** 2, axis=0)
    print(mu2, var2)"""

    # Random Initialization
    #Mu = (np.random.random_sample([clusterNum, data.shape[1]]) * 20).tolist()
    #Var = (np.random.random_sample([clusterNum, data.shape[1]]) * 5).tolist()
    #Pi = [1/clusterNum] * clusterNum

    # Manual Initialization
    # Mu = [[10, 0], [15, 50], [8, 30]]
    # Var = [[5, 10], [5, 10], [10, 10]]
    # Pi = [1/clusterNum] * clusterNum

    # Kmeans Initialization
    Mu, Var, Pi = kmeansInit(data, clusterNum)

    W = np.ones((data.shape[0], clusterNum)) / clusterNum
    loglh = []

    iterNum = 100
    Epsilon = 1e-8

    for _ in range(iterNum):
        if (time.time() - start_time > 10.0):
            print("Result log-likelihood (terminate without final result: 10 sec limit): ", lnl[-1])
            print("Result Mean (terminate without final result: 10 sec limit): ", Mu)
            print("Result Variance (terminate without final result: 10 sec limit): ", Var)
            break
        loglh.append(logLH(data, Pi, Mu, Var))
        #print("through")
        W = updateW(data, Mu, Var, Pi)
        Pi = updatePi(W)
        Mu = updateMu(data, W)
        Var = updateVar(data, Mu, W)
        try:
            if abs(loglh[-1]-loglh[-2]) < Epsilon:
                break
        except: pass

    print("Mu: ", Mu)
    print("Var: ", Var)
    #print("length: ", len(loglh))
    print("Log-likelihood: ", loglh[-1])
    print("BIC:", -2 * loglh[-1] * N + np.log(N)*(dim * clusterNum))
    print("time:", time.time() - start_time)
    

    #plot_clusters(data, Mu, Var, clusterNum)
    plt.figure()
    plt.plot(loglh)
    plt.show()



if __name__ == "__main__":
    main()