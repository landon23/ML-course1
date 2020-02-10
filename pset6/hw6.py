import numpy as np
from sklearn.svm._base import _fit_liblinear as libLinear
import scipy.sparse as sp
from scipy.optimize import minimize



class binLogistic:

    def __init__(self):
        pass

    def loadData(self, data, keys):
        self.xTrain = data[keys[0]]
        self.xTest = data[keys[1]]
        self.yTrain = data[keys[2]].flatten()
        self.yTest = data[keys[3]].flatten()
        if len(keys) ==6:
            self.xVal = data[keys[4]]
            self.yVal = data[keys[5]]
        else:
            self.xVal = data[keys[1]]
            self.yVal = data[keys[3]]
        if sp.issparse(self.xTrain):
            self.xTrain = self.xTrain.asformat('csr')
            self.xTest = self.xTest.asformat('csr')
            self.xVal = self.xVal.asformat('csr')
        else:
            self.xTrain = np.asarray(self.xTrain, order="C")



    def fit(self, penalty='l1', C=1.0, tol=0.000000001):
        self.w, self.b, n_iter = libLinear(self.xTrain, self.yTrain, C, False, 1.0, 'balanced', penalty, False, 0, 500, tol)
        #print(n_iter)

    def predict(self, x):
        A = x.dot(self.w.transpose())+self.b
        I = np.where(A>0)
        y = np.zeros((np.shape(x)[0], 1))-1
        y[I]=1
        return y.flatten()

    def acc(self, x, yActual):
        return np.sum(yActual.flatten() == self.predict(x)) / len(yActual.flatten())

class binLogSGD:
    def __init__(self):
        pass
    def loadData(self, data, keys):
        self.xTrain = data[keys[0]]
        self.yTrain = data[keys[1]].flatten()
        self.xTest = data[keys[2]]
        self.yTest = data[keys[3]].flatten()

    def fitLBFGS(self):
        x = self.xTrain
        y = self.yTrain

        init = np.zeros(1 + x.shape[1])  # + 0.0000001
        self.optimizationResult = minimize(logLoss, x0=init, args=(x, y), method='L-BFGS-B', jac=logGrad,
                                               tol=0.0001)
        self.w = self.optimizationResult.x
        self.iter = self.optimizationResult.nfev

    def fitSGD(self, batches=100, epochs=30, steps = []):
        n = self.xTrain.shape[0]

        w = np.zeros(1+self.xTrain.shape[1])
        self.testAcc = np.zeros(epochs)
        self.trainAcc = np.zeros(epochs)
        if len(steps)==0:
            steps = np.power(10.0 + np.arange(epochs), -0.51)
        elif len(steps) < epochs:
            steps2 = np.zeros(epochs)
            steps2[0:len(steps)]= steps
            steps2[len(steps):epochs] = steps[len(steps)-1]
            steps = steps2

        for j in range(epochs):
            I = np.random.permutation(n)
            x = self.xTrain[I, :]
            y = self.yTrain[I]
            eta = steps[j]
            #print('Epoch:', j)
            for i in range(batches):
                a = int(i*n / batches)
                b = int((i+1)*n / batches)
                xx = x [range(a, b), :]
                yy = y [ range(a, b)]
                w = w - eta* logGrad(w, xx, yy) /batches
            self.w = w
            self.testAcc[j] = self.acc(self.xTest, self.yTest)
            self.trainAcc[j] = self.acc(self.xTrain, self.yTrain)
        #self.w = w



    def predict(self, x):
        l = len(self.w)
        w = self.w[1:l]
        b = self.w[0]
        A = x.dot(w) + b
        I = np.where(A > 0)
        y = np.zeros((np.shape(x)[0], 1)) - 1
        y[I] = 1
        return y.flatten()

    def acc(self, x, yActual):
        return np.sum(yActual.flatten() == self.predict(x)) / len(yActual.flatten())


class binHuber:
    def __init__(self):
        pass

    def loadData(self, data, keys):
        self.xTrain = data[keys[0]]
        self.xTest = data[keys[1]]
        self.yTrain = data[keys[2]].flatten()
        self.yTest = data[keys[3]].flatten()
        if len(keys) ==6:
            self.xVal = data[keys[4]]
            self.yVal = data[keys[5]]
        else:
            self.xVal = data[keys[1]]
            self.yVal = data[keys[3]]
        if sp.issparse(self.xTrain):
            self.xTrain = self.xTrain.asformat('csr')
            self.xTest = self.xTest.asformat('csr')
            self.xVal = self.xVal.asformat('csr')

    def fit(self, C, delta):

        x = self.xTrain
        y = self.yTrain
        #delta = self.delta
        #C = self.C
        init = np.zeros(1+ x.shape[1]) #+ 0.0000001
        self.optimizationResult = minimize(objective, x0=init, args=(x, y, delta, C), method='L-BFGS-B', jac=grad, tol=0.0001)
        self.w = self.optimizationResult.x
        #print('Fitted')

    def predict(self, x):
        l = len(self.w)
        w = self.w[1:l]
        b= self.w[0]
        A = x.dot(w)+b
        I = np.where(A>0)
        y = np.zeros((np.shape(x)[0], 1))-1
        y[I]=1
        return y.flatten()

    def acc(self, x, yActual):
        return np.sum(yActual.flatten() == self.predict(x)) / len(yActual.flatten())


def huber(w, delta):
    I = np.where(np.abs(w) > delta)
    J = np.where(np.abs(w) <= delta)
    total = np.sum(w[J]*w[J]/2)
    total = total + np.sum(delta*np.abs(w[I])-delta*delta/2)
    return total

def huberGrad(w, delta):
    I = np.where(np.abs(w) > delta)
    J = np.where(np.abs(w) <= delta)
    grad = np.zeros(w.shape)
    grad[J] = w[J]
    grad[I] = delta*(2*np.heaviside(w[I], 0)-1)
    return grad

def logLoss(w, x, y):
    l = len(w)
    return np.sum(np.log(1+np.exp(-y*(x.dot(w[1:l])+w[0]))))

def logGrad(w, x, y):
    l = len(w)
    grad1= x.transpose().dot((-y ) / (1+ np.exp(y*(x.dot(w[1:l])+w[0]))))
    grad2= np.sum(- y / (1+np.exp(y*(x.dot(w[1:l])+w[0]))))
    return np.append(grad2, grad1)

def objective(w, x, y, delta, C):
    return huber(w, delta)+C* logLoss(w, x, y)

def grad(w, x, y, delta, C):
    return huberGrad(w, delta)+C*logGrad(w, x, y)

def ccheck():
    return 2



