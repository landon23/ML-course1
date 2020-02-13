import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class kMeans:

    def __init__(self):
        pass

    def fit(self, X, k, init=[]):

        if len(init)==0:
            I = np.random.permutation(X.shape[0])[0:k]
            mu = X[I, :]

        else:
            mu = init

        cont = True
        iter = 0
        self.J = []
        while cont:

            if iter > 300:
                cont = False
                print('Maximum iterations reached!')
            dist = cdist(X, mu, 'sqeuclidean')

            clusterNew = np.argmin(dist, axis=1)

            mins = dist[np.arange(dist.shape[0]), clusterNew]
            self.J.append(np.sum(mins)/X.shape[0])


            for i in range(k):
                I = np.where(clusterNew == i)
                qq = np.mean(X[I], axis=0)
                mu[i, :] = qq
            if iter >0:
                if np.sum(clusterOld == clusterNew)==len(clusterNew):
                    cont = False

            clusterOld = clusterNew.copy()




            iter = iter+1

        self.mu = mu
        self.iter = iter

        self.clusters = clusterNew



class bernoulliEM:

    def __init__(self):
        pass

    def fit(self, X, k):
        n = X.shape[0]
        d = X.shape[1]
        z = np.zeros((n, k))
        pi = np.zeros(k)+1/k
        mu = np.random.uniform(0.1, 0.9, (k, d))
        self.obj = []

        cont = True
        iter = 0
        while cont:
            if iter > 100:
                cont = False
            #E step
            tol = 0.00000001
            mu = np.maximum(mu, tol)
            mu = np.minimum(mu, 1-tol)
            pi = np.maximum(pi, tol)
            logp = X.dot(np.log(mu.transpose())) + (1-X).dot(np.log(1-mu.transpose()))
            a = np.exp(logp.transpose() - np.max(logp, axis=1)).transpose()*pi
            #print(np.min(np.sum(a, axis=1)))

            z = (a.transpose() / np.sum(a, axis=1)).transpose()

            #M step

            muOld = mu.copy()
            piOld = pi.copy()

            pi = np.sum(z, axis=0) / n
            #print(np.min(pi))

            mu = (z.transpose().dot(X)).transpose() / np.sum(z, axis=0)
            mu = mu.transpose()


            if (np.max(np.abs(mu-muOld)) < 0.0000001) and (np.max(np.abs(pi-piOld)) < 0.0000001):
                cont= False
            iter = iter +1
            #print(iter)
            mu = np.maximum(mu, tol)
            mu = np.minimum(mu, 1 - tol)
            pi = np.maximum(pi, tol)
            pp = X.dot(np.log(mu.transpose())) + (1-X).dot(np.log(1-mu.transpose()))
            j = np.max(pp, axis=1)
            pp = np.sum(np.exp(pp.transpose()-j).transpose()*pi)

            self.obj.append(pp+np.sum(j))


        self.iter = iter

        self.mu = mu
        self.pi = pi

    def prob(self, X):
        mu = self.mu
        pi = self.pi
        tol = 0.00000001
        mu = np.maximum(mu, tol)
        mu = np.minimum(mu, 1 - tol)
        pi = np.maximum(pi, tol)
        ll = X.dot(np.log(mu.transpose()))+(1-X).dot(np.log(1-mu.transpose()))
        jj = np.max(ll, axis=1)
        ll = ll.transpose() - jj
        ll = ll.transpose()
        pp = np.exp(ll)*pi
        pp = pp.transpose() / np.sum(pp, axis=1)
        return pp.transpose()



def printLabels(pp, labels, k):
    for i in range(k):
        prob = pp[:, i]
        I = np.argpartition(prob, -5)[-5:]
        lab =labels[I]
        stri = ''
        for j in range(5):
            stri = stri+lab[j, 0][0]+', '
        print('Top words associated with cluster '+str(i)+': '+stri)

def printFeatures(mu, labels, k):
    for i in range(k):
        muu = mu[i, :]
        I = np.argpartition(muu, -5)[-5:]
        lab = labels[I]
        stri= ''
        for j in range(5):
            stri = stri+lab[j, 0][0]+','
        print('Top features associated with cluster '+str(i)+': '+stri)



def randInd(y1, y2, k1, k2=0):
    n = len(y1)

    c = 0
    d = 0

    if k2 ==0:
        k2 = k1


    for i in range(k1):
        I = np.where(y1 == i)
        hold = 0
        for j in range(k2):
            u = y2[I[0]]
            u = (u ==j)
            hold  = hold+ np.sum(u)*(np.sum(u)-1)/2

        m = len(I[0])
        c = c + m*(m-1)/2 - hold

    for i in range(k2):
        I = np.where(y2==i)
        hold = 0
        for j in range(k1):
            u = y1[I[0]]
            u = (u == j)
            hold = hold+np.sum(u)*(np.sum(u)-1)/2

        m = len(I[0])
        d = d+ m*(m-1)/2 - hold

    return 1 - (c+d)*2 / (n*(n-1))




def plot(x):
    #Function for plotting grey-scale handwritten digits.
    e = np.array([1, 1, 1])
    pic = np.tensordot(x.reshape([28, 28]), e, axes=0)
    plt.imshow(255 - pic)
    plt.show()

def plot2(x1, x2):
    #Function for plotting grey-scale handwritten digits.
    e = np.array([1, 1, 1])
    x1 = x1 - np.min(x1)
    x1 = x1 / (np.max(x1))
    x1 = 255*x1
    x1 = x1.astype(int)
    x2 = x2 - np.min(x2)
    x2 = x2 / (np.max(x2))
    x2 = 255 * x2
    x2 = x2.astype(int)
    pic1 = np.tensordot(x1.reshape([28, 28]), e, axes=0)
    pic2 = np.tensordot(x2.reshape([28, 28]), e, axes=0)
    #plt.imshow(255 - pic)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(255 - pic1)
    ax[1].imshow(255 - pic2)
    plt.show()

def plot5(x):
    e = np.array([1, 1, 1])
    x = x - np.min(x)
    x = x /(np.max(x))
    x = 255*x
    x = x.astype(int)
    pic={}
    for i in range(5):
        pic[str(i)] = np.tensordot(x[i, :].reshape([28, 28]), e, axes=0)
    fig, ax = plt.subplots(1, 5, figsize=(15, 4.8))
    for i in range(5):
        ax[i].imshow(255-pic[str(i)])
    plt.show()