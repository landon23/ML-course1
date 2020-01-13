import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class logisticRegression:
    def __init__(self, k=2, scale= True):
        self.k = k
        self.scaleVar = scale #whether to scale the data to fit between [-1, 1]

    def loadData(self, data, keys):
        if len(keys)==4:
            keys.append(keys[2])
            keys.append(keys[3])
        self.train = data[keys[0]]
        self.trainLabels = data[keys[1]]
        self.test = data[keys[2]]
        self.testLabels = data[keys[3]]
        self.dev = data[keys[4]]
        self.devLabels = data[keys[5]]
        labels = np.arange(self.k)+1
        if self.trainLabels.shape[1] == 1:
            self.yTrain = (self.trainLabels == labels)
            self.yTest = (self.testLabels == labels)
            self.yDev = (self.devLabels == labels)
        else:
            self.yTrain = self.trainLabels
            self.yTest = self.testLabels
            self.yDev = self.devLabels
        if self.scaleVar:
            self.scale()

    def scale(self):
        A = np.max(self.train, axis=0)
        B = np.min(self.train, axis=0)
        self.train = self.scaleX(self.train, A, B)
        self.test = self.scaleX(self.test, A, B)
        self.dev = self.scaleX(self.dev, A, B)

    def scaleX(self, x, A, B):
        x = x- (A+B)/2
        return 2*x/(B-A)

    def makeFeatures(self, method = 0):
        #make linear or quadratic features
        [self.xTrain, self.xTest, self.xDev] = [self.mF(self.train, method), self.mF(self.test, method), self.mF(self.dev, method)]

    def mF(self, x, method):
        #makes quadratic features from multidimensional data
        if method ==2:
            hold = np.einsum('ij, ik -> jki', x, x)
            I = np.triu_indices(x.shape[1])
            xx = hold[I]
            xx = xx.transpose()
            return np.hstack((np.ones((x.shape[0], 1)), x, xx))

        elif method == 1:
            return np.hstack((np.ones((x.shape[0], 1)), x, np.power(x, 2)))

        else:
            return np.hstack((np.ones((x.shape[0], 1)), x))

    def fit(self, alpha):
        x = self.xTrain
        y = self.yTrain
        k = self.k
        init = np.zeros(k*x.shape[1])+0.0000001
        self.optimizationResult = minimize(self.objective, x0 = init, args = (x, y, alpha, k), method='Newton-CG', jac=self.gradient)
        self.w = self.optimizationResult.x

    @staticmethod
    def objective(w, x, y, alpha, k):
        #calculates the objective function for Logistic regression
        f = 0

        w = np.reshape(w, (k, x.shape[1]))
        f = f + alpha*np.sum(np.power(w, 2))/2
        f = f - np.trace(np.matmul(np.matmul(y, w), x.transpose()))
        f = f+np.sum(np.log(np.sum(np.exp(np.matmul(w, x.transpose())), axis=0)))
        return f

    @staticmethod
    def gradient(w, x, y, alpha, k):
        #Calculates the gradient for Logistic regression
        grad = 0

        w = np.reshape(w, (k, x.shape[1]))
        grad = grad + alpha*w
        grad = grad - np.matmul(y.transpose(), x)
        a = np.matmul(w, x.transpose())
        a = a - np.max(a, axis=0)
        A = np.exp(a) / np.sum(np.exp(a), axis=0)
        grad = grad + np.matmul(A, x)
        return np.reshape(grad, (w.shape[0]*w.shape[1]))

def plotDecision(w, x, y):
    # A function which plots decision boundaries for two dimensional data for linear classifiers.  Can do 2 or 3 classes.

    A = np.min(x[:, 0])
    B = np.max(x[:, 0])
    C = np.min(x[:, 1])
    D = np.max(x[:, 1])
    ymax = np.zeros(1000) + D + 1
    ymin = np.zeros(1000) + C - 1
    grid = np.linspace(A, B, 1000)
    yy = np.argmax(y, axis=1)

    if w.shape[1] ==2:
        q = w[:, 1]-w[:, 0]

        class0 = np.where(yy==0)
        class1 = np.where(yy==1)

        c = q[0]
        a = q[1]
        b = q[2]
        yline = plotLine(grid, a, b, c)
        if b >0:
            plt.fill_between(grid, yline, ymax, where = (yline < ymax), edgecolor='blue', hatch='x')
        elif b<0:
            plt.fill_between(grid, yline, ymin, where = (yline > ymin), facecolor='blue', hatch='x')

        plt.scatter(x[:, 0][class0], x[:, 1][class0], color='red', label='class 0', edgecolors='black')
        plt.scatter(x[:, 0][class1], x[:, 1][class1], color='blue', label='class 1', edgecolors='black')
    if w.shape[1]==3:
        class0 = np.where(yy==0)
        class1 = np.where(yy==1)
        class2 = np.where(yy==2)
        q10 = w[:, 1] - w[:, 0]
        q20 = w[:, 2] - w[:, 0]
        q21 = w[:, 2] - w[:, 1]

        y21 = plotLine(grid, q21[1], q21[2], q21[0])
        y20 = plotLine(grid, q20[1], q20[2], q20[0])

        if (q21[2] > 0) & (q20[2] > 0):
            #above both lines
            yline = np.maximum(y21, y20)
            plt.fill_between(grid, yline, ymax, where=(yline < ymax),  edgecolor='green', hatch='x', facecolor='green')

        elif (q21[2] > 0) & ( q20[2] < 0):
            #above y21, below y20
            plt.fill_between(grid, y21, y20, where=(y21 < y20), edgecolor='green', hatch='x', facecolor='green')
        elif (q21[2] < 0) & (q20[2] > 0):
            plt.fill_between(grid, y21, y20, where=(y21 > y20), edgecolor='green', hatch='x', facecolor='green')
        else:
            #below both lines
            yline = np.minimum(y21, y20)
            plt.fill_between(grid, yline, ymin, where = (yline > ymin), edgecolor='green', hatch='x', facecolor='green')

        y10 = plotLine(grid, q10[1], q10[2], q10[0])

        if (q21[2] < 0) & (q10[2] > 0):
            #above both lines
            yline = np.maximum(y21, y10)
            plt.fill_between(grid, yline, ymax, where=(yline < ymax), edgecolor='blue', hatch='x')

        elif (q21[2] < 0) & ( q10[2] < 0):
            #above y21, below y10
            plt.fill_between(grid, y21, y10, where=(y21 < y10), edgecolor='blue', hatch='x')
        elif (q21[2] > 0) & (q10[2] > 0):
            plt.fill_between(grid, y21, y10, where=(y21 > y10), edgecolor='blue', hatch='x')
        else:
            #below both lines
            yline = np.minimum(y21, y10)
            plt.fill_between(grid, yline, ymin, where=(yline > ymin), edgecolor='blue', hatch='x')

        plt.scatter(x[:, 0][class0], x[:, 1][class0], color='red', label='class 0', edgecolors='black')
        plt.scatter(x[:, 0][class1], x[:, 1][class1], color='blue', label='class 1', edgecolors='black')
        plt.scatter(x[:, 0][class2], x[:, 1][class2], color='green', label='class 2', edgecolors='black')

    plt.show()

def plotLine(x, a, b, c):
    return (- c - a*x)/b
def validationPlot(optR, classifier, logAlpha, title):
    n = len(optR)
    logL = [optR[i].fun for i in range(n)]
    acc = np.zeros(n)
    x = classifier.xTrain
    y = classifier.trainLabels.flatten()
    for i in range(10):
        w = np.reshape(optR[i].x, (2, -1))
        p = np.argmax(np.matmul(x, w.transpose()), axis=1) + 1
        acc[i] = np.sum(p.flatten() == y) / len(y)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(-logAlpha, logL)
    ax[1].plot(-logAlpha, acc)
    ax[0].set_xlabel('-log alpha')
    ax[0].set_ylabel('Log likelihood')
    ax[0].set_title('Log likelihood')
    ax[1].set_xlabel('-log alpha')
    ax[1].set_ylabel('training accuracy')
    ax[1].set_title('Training accuracy')
    fig.tight_layout(pad=1.0)
    fig.set_figwidth(12)
    fig.suptitle(title)
    plt.show()

class optR:
    def __init__(self, fun, x):
        self.fun = fun
        self.x = x
