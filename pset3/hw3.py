import numpy as np
import matplotlib.pyplot as plt

#This is a Gaussian classifier which considers each feature an independent Gaussian random variable with a mean and variance that are fitted from the MLE.
class gaussianClassifier:
    def __init__(self, data, k=2):
        self.k = k #number of classes, default 2
        self.xTrain = data['train']
        self.yTrain = data['trainLabels'] #yTrain and yTest are N-vectors containing the class label
        self.xTest = data['test']
        self.yTest = data['testLabels']
        self.labels = np.arange(k)
        self.lTrain = (self.yTrain == self.labels)+0 #lTrain, lTest are N x k matrices
        self.lTest = (self.yTest == self.labels)+0
        self.threshold = 0.0

    def fit(self):
        #calculates MLEs
        self.pi = np.sum(self.lTrain, axis=0) / self.xTrain.shape[0]
        self.mu = np.matmul(self.lTrain.transpose(),self.xTrain).transpose() / np.sum(self.lTrain, axis = 0)
        self.mu = self.mu.transpose()
        hold = np.power(self.xTrain - np.matmul(self.lTrain, self.mu), 2)
        self.sigmaSq = np.matmul(self.lTrain.transpose(),hold).transpose() / np.sum(self.lTrain, axis = 0)
        self.sigmaSq = np.maximum(self.sigmaSq, self.threshold) #threshold the variance to not be 0.
        self.sigmaSq = self.sigmaSq.transpose()

    def calcCondLL(self, x, mu, sigma):
        #calculate log-likelihoods
        xn = np.einsum('ij, k', x, np.ones(self.k))
        mun = np.einsum('i, kj', np.ones(x.shape[0]), mu)
        t = xn - mun
        w = np.einsum('ijk,ijk,kj -> ik', t, t, 1 / sigma)
        logSig = np.sum(np.log(sigma), axis=1)
        return 0 - w /2 - logSig / 2

    def predict(self, x, weights=1):
        #Predicts class of test samples x by finding class with largest loglikelihood, accounting for weighting.
        if str(type(weights))!= str(type(1)):
            weights = weights.flatten()
        ll = self.calcCondLL(x, self.mu, self.sigmaSq)
        ll = ll + np.log(weights)
        return np.argmax(ll, axis=1)

    def bayesPredict(self, x):
        return self.predict(x, self.pi)


    def accuracy(self, y1, y2):
        return np.sum(y1.flatten()==y2.flatten()) / len(y1)

    def roc(self):
        #Plots a ROC curve using test dataset.
        ll = self.calcCondLL(self.xTest, self.mu, self.sigmaSq)
        y = self.yTest
        plotROC(ll, y)

    def reportRates(self, weights):
        #Reports TPR and FPR on test dataset using specified weights for the prior/loss function.
        y = self.predict(self.xTest, weights).flatten()
        p = np.sum(self.yTest)
        n = len(y)-p
        tpr = np.sum((y==1) & (self.yTest.flatten() == 1)) / p
        fpr = np.sum((y == 1) & ( self.yTest.flatten() == 0 )) / n

        print('TPR: '+'{:.4f}'.format(tpr), 'FPR: '+'{:.4f}'.format(fpr))








def roc(ll, y):
    #Calculates a ROC curve
    y = y.flatten()
    l = ll[:,1]-ll[:,0]
    I = np.argsort(l)
    p = np.sum(y)
    n = len(y)-p
    tp = np.sum(y)
    fp = len(y) - tp
    graph = np.zeros([len(y), 2])
    for i in range(len(y)):
        if y[I[i]] == 1:
            tp = tp -1
        else:
            fp = fp -1
        graph[i, :] = [fp/n, tp/p]
    return graph

def plotROC(ll, y):
    #given a matrix of loglikelihoods for a binary classification problem and true labels, plots a ROC curve
    graph = roc(ll, y)
    plt.plot(graph[:, 0], graph[:, 1])
    plt.plot([0, 1], [0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.show()



def plot(x):
    #Function for plotting grey-scale handwritten digits.
    e = np.array([1, 1, 1])
    x = x - np.min(x)
    x = x / (np.max(x))
    x = 255*x
    x = x.astype(int)
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


