import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

class linReg:
    def __init__(self):
        pass

    def loadData(self, data, keys):
        self.featTrain = data[keys[0]]
        self.yTrain = data[keys[1]]
        self.featTest = data[keys[2]]
        self.yTest = data[keys[3]]
        self.xTrain = self.featTrain
        self.xTest = self.featTest
        self.bTrain = np.ones((self.xTrain.shape[0], 1))
        self.bTest = np.ones((self.xTest.shape[0], 1))
        self.featGrid = np.linspace(np.min(self.xTrain), np.max(self.xTrain), 200)
        self.featGrid = np.reshape(self.featGrid, (-1, 1))
        self.xGrid = self.featGrid

    def makePhi(self):
        self.phiTrain = np.hstack((self.bTrain, self.xTrain))
        self.phiTest = np.hstack((self.bTest, self.xTest))


    def polyBasis(self, x, k):
        orig = x
        if k > 1:
            for i in range(2, k+1):
                x = np.hstack((x, np.power(orig, i)))
        return x

    def radialBasis(self, x, L):
        mu = np.linspace(-1, 1, num=L)
        sigmaSq = np.power( 2 / (L-1), 2.0)
        x = x.flatten()
        A = np.einsum('i, j', x, np.ones(len(mu))) - np.einsum('i, j', np.ones(len(x)), mu)
        return np.exp(0 - np.power(A, 2.0) / (2 * sigmaSq))

    def fitReg(self, delta):
        phi = np.hstack((self.bTrain, self.xTrain))
        self.phi = phi
        self.w = self.getRegParams(phi, self.yTrain, delta)

    def getRegParams(self, phi, y, delta):
        return linalg.solve( np.matmul(phi.transpose(), phi)+delta*np.identity(phi.shape[1]), phi.transpose().dot(y), assume_a='pos')



    def radialLinReg(self, L):
        self.xTrain = self.featTrain
        self.xTest = self.featTest
        self.xGrid = self.featGrid
        self.scale()
        self.xTrain = self.radialBasis(self.xTrain, L)
        self.xTest = self.radialBasis(self.xTest, L)
        self.xGrid = self.radialBasis(self.xGrid, L)
        #self.scale()

    def radialRange(self, L):
        trainE = np.zeros(L-1)
        testE = np.zeros(L-1)
        for i in range(2, L+1):
            self.radialLinReg(i)
            self.fit()
            trainP = self.predict(self.xTrain)
            testP = self.predict(self.xTest)
            trainE[i-2]=np.linalg.norm(trainP-self.yTrain) / np.sqrt(len(self.yTrain))
            testE[i-2] = np.linalg.norm(testP - self.yTest) / np.sqrt(len(self.yTest))
        return [trainE, testE]


    def polyLinReg(self, k):
        #self.scale()
        self.xTrain = self.featTrain
        self.xTest = self.featTest
        self.xGrid = self.featGrid
        self.scale()
        self.xTrain = self.polyBasis(self.xTrain, k)
        self.xTest = self.polyBasis(self.xTest, k)
        self.xGrid = self.polyBasis(self.xGrid, k)
        #self.scale()

    def scale(self):
        B = np.max(self.xTrain, axis=0)
        A = np.min(self.xTrain, axis=0)
        self.xTrain = self.xTrain - (A+B)/2
        self.xTrain = 2 * self.xTrain / (B-A)
        self.xTest = self.xTest - (A+B)/2
        self.xTest = 2* self.xTest / (B-A)
        self.xGrid = self.xGrid  - (A+B)/2
        self.xGrid = 2* self.xGrid / (B-A)

    def fit(self):
        phi = np.hstack((self.bTrain, self.xTrain))
        self.w = self.getParams(phi, self.yTrain) #np.linalg.solve(np.matmul(phi.transpose(), phi), phi.transpose().dot(self.yTrain))

    def fitNo(self):
        self.w = self.getParams(self.xTrain, self.yTrain)

    def plotNo(self):
        x = self.xTest
        y = x.dot(self.w)

        plt.scatter(self.featTest, y, label='Prediction')
        plt.scatter(self.featTest, self.yTest, label='Test data')
        plt.legend()
        plt.show()

    def plotNoTrain(self):
        x = self.xTrain
        y = x.dot(self.w)
        xx = np.linspace(-1, 1, 1000)
        x = np.reshape(xx, (-1, 1))
        x = self.radialBasis(x, 25)
        y = x.dot(self.w)

        self.xTrain = self.featTrain
        self.scale()
        plt.scatter(xx, y, label='Prediction')
        plt.scatter(self.xTrain[:, 0], self.yTrain, label='Train data')
        plt.legend()
        plt.show()


    def plotTrain(self):
        x = np.hstack((self.bTrain, self.xTrain))
        y = x.dot(self.w)

        plt.scatter(self.featTrain, y, label='Prediction')
        plt.scatter(self.featTrain, self.yTrain, label='Train data')
        plt.legend()
        plt.show()


    def plot(self):
        x = np.hstack((self.bTest, self.xTest))
        y = x.dot(self.w)

        plt.scatter(self.featTest, y, label='Prediction')
        plt.scatter(self.featTest, self.yTest, label='Test data')
        plt.legend()
        plt.show()

    def getParams(self, phi, y):
        #return np.linalg.lstsq(np.matmul(phi.transpose(), phi), phi.transpose().dot(y), rcond=None)[0]

        #return np.linalg.solve(np.matmul(phi.transpose(), phi), phi.transpose().dot(y))

        #return linalg.solve(np.matmul(phi.transpose(), phi) + 0.0000000001 *np.identity(phi.shape[1]), phi.transpose().dot(y), assume_a='pos')

        #return linalg.lstsq(np.matmul(phi.transpose(), phi), phi.transpose().dot(y))[0]
        return linalg.lstsq(phi, y)[0]

    def predict(self, x):
        phi = np.hstack((np.ones((x.shape[0], 1)), x))
        return self.predict_func(phi, self.w)

    def predict_func(self, phi, w):
        return phi.dot(w)

    def errorRange(self, L):
        self.xTest = self.featTest
        self.xTrain = self.featTrain
        self.polyLinReg(L)
        trainE = np.zeros(L)
        testE = np.zeros(L)
        for i in range(1, L+1):
            phiTrain = self.xTrain[:,0:i*self.featTrain.shape[1]]
            #print(phiTrain.shape)
            phiTrain = np.hstack((self.bTrain, phiTrain))
            phiTest = self.xTest[:,0:i*self.featTrain.shape[1]]
            phiTest = np.hstack((self.bTest, phiTest))
            w = self.getParams(phiTrain, self.yTrain)
            trainP = self.predict_func(phiTrain, w)
            testP = self.predict_func(phiTest, w)
            trainE[i-1] = np.linalg.norm(self.yTrain-trainP)
            testE[i-1] = np.linalg.norm(self.yTest-testP)
        trainE = trainE / np.sqrt(len(self.yTrain))
        testE = testE / np.sqrt(len(self.yTest))
        return [trainE, testE]

    def errorRegRange(self, deltas, L, basis = 'poly'):
        self.xTest = self.featTest
        self.xTrain = self.featTrain
        trainE = np.zeros(len(deltas))
        testE = np.zeros(len(deltas))
        if basis == 'poly':
            self.scale()
            self.polyLinReg(L)
        elif basis == 'radial':
            self.radialLinReg(L)
        for i in range(len(deltas)):
            delta = deltas[i]
            self.fitReg(delta)

            trainP = self.predict(self.xTrain)
            testP = self.predict(self.xTest)

            trainE[i] = np.linalg.norm(self.yTrain - trainP) / np.sqrt(len(self.yTrain))
            testE[i] = np.linalg.norm(self.yTest - testP) / np.sqrt(len(self.yTest))

        return [trainE, testE]






















