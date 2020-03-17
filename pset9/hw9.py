import numpy as np
from scipy.linalg import solve
from scipy.linalg import lstsq
from scipy.linalg import eigh
import pickle

class hMM:

    def __init__(self, K):
        self.K = K

    def fit(self, X, d, max_iter = 500):
        n = len(X)
        K = self.K
        self.d = d
        cont = True
        iter = 0
        #initialize A, mu, pi
        pi = np.zeros(K) + 1/K
        mu = np.random.uniform(low = 0.25, high = 0.75, size=(K, d))
        mu = (mu.transpose() / np.sum(mu, axis=1)).transpose()

        A = np.random.uniform(low=0.25, high = 0.75, size=(K, K))
        A = (A / np.sum(A, axis=0)).transpose()
        prob = 0.0
        self.ll = []

        XX = np.zeros((n, d))
        XX[np.arange(n), X]=1

        while cont:
            iter = iter +1
            #print('iteration:', iter)
            if iter >max_iter:
                cont = False
                print('Max iterations reached')
            #E step: calculate alphahat, betahat, c_n:
            alphaHat = np.zeros((n, K))
            c = np.zeros(n)


            alpha = pi*self.emission(X[0], mu)
            c[0] = np.sum(alpha)
            alphaHat[0, :] = alpha / c[0]

            for i in range(1, n):
                alphaNm = alphaHat[i-1, :]
                alpha = self.emission(X[i], mu)*(A.transpose().dot(alphaNm))
                c[i] = np.sum(alpha)
                alphaHat[i, :] = alpha / c[i]

            betaHat = np.zeros((n, K))
            betaHat[n-1, :] = np.ones(K)
            for i in range(n-2, -1, -1):
                betaOld = betaHat[i+1, :]
                beta = A.dot(betaOld*self.emission(X[i+1], mu))
                betaHat[i, :] = beta / c[i+1]
            probOld = prob
            prob = -np.sum(np.log(c))
            self.ll.append(prob)
            if np.abs(probOld -prob) < 0.00000001 and iter >1:
                cont = False
                print('Tolerance reached!')


            self.alpha = alphaHat
            self.beta = betaHat
            self.c = c

            #make gamma, xi
            gamma = alphaHat*betaHat
            xi = np.zeros((n-1, K, K))
            p = mu[:, X]
            xi = np.einsum('i, ij, ki, jk, ik ->ijk', c[1:n], alphaHat[0:n-1, :], p[:, 1:n], A, betaHat[1:n, :])
            #xi[i, j, k] = xi (z_i = j, z_{i+1} = k )

            #cc= c[1:n]
            #aa = alpha[0:n-1, :]
            #pp = p[:, 1:n]
            #bb = beta[1:n, :]
            #xi = np.einsum('i, ij, ki, jk, ik -> ijk', cc, aa, pp, A, bb)

            self.gamma = gamma
            self.xi = xi

            #Mstep:
            #pi = (gamma[0, :] ) / np.sum(gamma[0, :])

            #pi with prior:
            a=0.000000001
            pi = (gamma[0, :] + a)
            pi = pi / np.sum(pi)


            #mu = gamma.transpose().dot(XX)
            #mu = (mu.transpose() / np.sum(gamma, axis=0)).transpose()

            #mu with prior:
            a=0.00000001
            mu = gamma.transpose().dot(XX) + a
            mu = (mu.transpose() / np.sum(mu, axis=1)).transpose()




            #A = np.sum(xi, axis=0)
            #A = (A.transpose() / np.sum(A, axis=1)).transpose()

            #A with prior:
            A = np.sum(xi, axis=0) + 0.0000001
            A = (A.transpose() / np.sum(A, axis=1)).transpose()



            self.A = A
            self.mu = mu
            self.pi = pi


        params={}
        params['A']=A
        params['mu']=mu
        params['pi']=pi
        self.params = params

    def marginalMissing(self, X, eps):
        d = self.d+1
        K = self.K
        muOld = self.mu
        mu = np.zeros((K, d))
        mu[:, 0:d-1]=muOld*(1-eps)
        mu[:, d-1]=eps
        n = len(X)
        mu = mu.transpose()/np.sum(mu, axis=1)
        mu = mu.transpose()
        A = self.A
        pi = self.pi
        alphaHat = np.zeros((n, K))
        c = np.zeros(n)

        alpha = pi * self.emission(X[0], mu)
        c[0] = np.sum(alpha)
        alphaHat[0, :] = alpha / c[0]

        for i in range(1, n):
            alphaNm = alphaHat[i - 1, :]
            alpha = self.emission(X[i], mu) * (A.transpose().dot(alphaNm))
            c[i] = np.sum(alpha)
            alphaHat[i, :] = alpha / c[i]

        betaHat = np.zeros((n, K))
        betaHat[n - 1, :] = np.ones(K)
        for i in range(n - 2, -1, -1):
            betaOld = betaHat[i + 1, :]
            beta = A.dot(betaOld * self.emission(X[i + 1], mu))
            betaHat[i, :] = beta / c[i + 1]

        gamma = alphaHat * betaHat
        xi = np.zeros((n - 1, K, K))
        p = mu[:, X]
        xi = np.einsum('i, ij, ki, jk, ik ->ijk', c[1:n], alphaHat[0:n - 1, :], p[:, 1:n], A, betaHat[1:n, :])

        return gamma, xi


    def predictMissing(self, X, eps):
        d = self.d
        gamma, xi = self.marginalMissing(X, eps)
        n = len(X)
        mu = self.mu
        out = X.copy()
        for i in range(n):
            if X[i]==d:
                out[i] = np.argmax((mu.transpose().dot(gamma[i, :])))
        return out



    def sample(self, m):
        z = np.zeros(m)
        z[0] = np.argmax(np.random.multinomial(1, self.pi))
        for i in range(1, m):
            pp = self.A[z[i-1].astype(int), :]
            z[i] = np.argmax(np.random.multinomial(1, pp))

        x = np.zeros(m)
        for i in range(m):
            pp = self.mu[z[i].astype(int), :]
            x[i] = np.argmax(np.random.multinomial(1, pp))

        return x

    def calcLL(self, X):
        d = self.d
        K = self.K
        n = len(X)
        A = self.A
        mu = self.mu
        pi = self.pi
        alphaHat = np.zeros((n, K))
        c = np.zeros(n)

        alpha = pi * self.emission(X[0], mu)
        c[0] = np.sum(alpha)
        alphaHat[0, :] = alpha / c[0]
        for i in range(1, n):
            alphaNm = alphaHat[i - 1, :]
            alpha = self.emission(X[i], mu) * (A.transpose().dot(alphaNm))
            c[i] = np.sum(alpha)
            alphaHat[i, :] = alpha / c[i]
        return -np.sum(np.log(c))

    def emission(self, x, mu):
        # returns the vector p(x | z) which has dimension K (x is a scalar)
        return mu[:, x]


def aic(logLL, K, d):
    return logLL - K*K-K*(d-1)

def bic(logLL, d, K, n):
    return logLL - (K*K+K*(d-1))*(np.log(n))/2


def num_to_word(x):
    vals = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z', ' ']
    str = ' '
    for i in range(len(x)):
        str = str+vals[x[i].astype(int)]
    return str

def generate(A, mu, m):
    x = [0]*m
    z = np.zeros(m)
    for i in range(1, m):
        z[i] = np.argmax(np.random.multinomial(1, A[z[i-1].astype(int), :]))
        x[i] = np.argmax(np.random.multinomial(1, mu[z[i].astype(int), :])).astype(int)
    return x

def deleteChars(X, d, eps):
    n = len(X)
    J = np.random.binomial(1, eps, size=n) #eps smaller means less Ones
    out = X.copy()
    for i in range(n):
        if J[i]==1:
            out[i]=d
    return out

def loadText(str):
    cont = True
    lis = []
    with open(str) as f:
        while cont:
            c = f.read(1)
            if not c:
                cont = False
            else:
                lis.append(c)

    chars = []
    vals = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z', ' ']

    for i in range(len(lis)):
        for j in range(len(vals)):
            if vals[j] == lis[i]:
                chars.append(j)
    return chars

class latentFactor:
    def __init__(self, K):
        self.K = K
        self.W = []
        self.m = []
        self.V = []


    def fit(self, X):
        n = X.shape[0]
        d = X.shape[1]
        R = 1- (X==0)
        K = self.K
        Z = np.zeros((n, K))
        W = np.random.uniform(-1, 1, size=(d, K))
        m = np.random.uniform(-1, 1, size=(d, ))
        V = 1 + np.random.uniform(-0.5, 0.5, size=(d,)) #this is Variance, not STD.
        cont = True
        iter = 0
        #E step
        while cont:
            iter = iter +1
            #print('Iteration: '+str(iter))
            if iter > 100:
                cont = False

            ###E STEP:
            for i in range(n):
                #C = (W.transpose().dot(np.diag(R[i, :]/V))).dot(W)
                #eigs = eigh(C)
                #print(eigs)
                A = (W.transpose().dot(np.diag(R[i, :]/V))).dot(W)+np.identity(K)
                #A = (A + A.transpose())/2
                Z[i, :] = solve(A, W.transpose().dot(R[i, :]*(X[i, :]-m)/V), sym_pos=True)
            
            ###M STEP:
            Zint = np.ones((n, K+1))
            Zint[:, 0:K]=Z
            #print('max Z:', np.max(np.abs(Zint)))

            for i in range(d):
                I = np.where(R[:, i]==1)[0]
                #Q = (Zint.transpose()*R[:, i]).transpose()
                #A = Q.transpose().dot(Q)
                #A = (A + A.transpose())/2
                #beta = solve(A, Q.transpose().dot(X[:, i]), sym_pos=True)
                Q = Zint[I, :]
                xx = X[I, i]

                #beta = lstsq(Q, X[:, i])[0]
                beta = lstsq(Q, xx)[0]
                #resid = np.sum(np.power(Q.dot(beta)-X[:, i], 2.0))/np.sum(R[:, i])
                resid = np.sum(np.power(Q.dot(beta)-xx, 2.0))/ np.sum(R[:, i])
                W[i, :]=beta[0:K]
                m[i]=beta[-1]
                #if resid < 0.00001:
                    #print('small residual')
                V[i]=np.max((resid, 0.000001))
            #print('Max W', np.max(np.abs(W)))

            #print('Max V', np.max(V))

            #return Zint
        self.Z = Z
        self.W = W
        self.m = m
        self.V = V

    def predict(self):
        Z = self.Z
        W = self.W
        m = self.m
        V = self.V
        x = Z.dot(W.transpose())+m
        x = np.maximum(x, 1.0)
        x = np.minimum(x, 5.0)

        return x

    def RMS(self, xTest):
        rTest = 1 - (xTest ==0)
        xPredict = self.predict()
        err = np.sum(np.power(rTest*(xTest-xPredict), 2.0)) / np.sum(rTest)
        return np.sqrt(err)

def RMS(xPredict, xTest):
    rTest = 1 - (xTest==0)
    err = rTest*(xTest-xPredict)
    err = np.sum(np.power(err, 2.0)) / np.sum(rTest)
    return np.sqrt(err)

class moviePCA:

    def __init__(self):
        pass
    def fit(self, X):
        r = 1 - (X==0)
        means = np.sum(X, axis=0)  / np.sum(r, axis=0)
        Xfill = X + (1-r)*means

        U, S, V = np.linalg.svd(Xfill)
        self.V = V
        self.Xfill = Xfill

    def predict(self, K):
        V = self.V
        vv = V[0:K, :]
        xPredict = (vv.transpose().dot(vv)).dot(self.Xfill.transpose())
        xPredict = xPredict.transpose()
        return xPredict

def loadMod():
    mod1 = hMM(1)
    mod60 = hMM(60)
    modB = hMM(20)
    params =pickle.load(open('params.pkl', 'rb'))
    par1 = params['1']
    par60 = params['60']
    parB = params['20']
    mod1.A = par1['A']
    mod1.mu = par1['mu']
    mod1.pi = par1['pi']
    mod1.d=27
    mod60.A = par60['A']
    mod60.mu = par60['mu']
    mod60.pi = par60['pi']
    mod60.d = 27
    modB.A = parB['A']
    modB.mu = parB['mu']
    modB.pi = parB['pi']
    modB.d = 27
    return mod1, modB, mod60

def predictAcc(original, predict, missing):
    n = len(original)
    corr = 0
    total = 0
    for i in range(n):
        if missing[i] ==27:
            total = total+1
            if original[i]==predict[i]:
                corr = corr+1
    return corr/total