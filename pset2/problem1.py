import numpy as np

#Classifier for spam detection using a Naive Bayes Bernoulli or Geometric generative model.
class spamClassifier:
    def __init__(self, data, threshold=0.001, method='Bernoulli'):
        #Load data
        self.xTrain = data['trainFeat']
        self.yTrain = data['trainLabels']
        self.xTest = data['testFeat']
        self.yTest = data['testLabels']
        self.xVal = data['valFeat']
        self.yVal = data['valLabels']
        self.threshold=threshold
        #Save whether using Bernoulli or Geometric generative model
        if method=='Bernoulli':
            self.method=1
        else:
            self.method=0
        self.zTrain = self.truncate(self.xTrain)
        self.zTest = self.truncate(self.xTest)
        self.zVal = self.truncate(self.xVal)
        self.fitted=0


    def truncate(self, x):
        # Thresholds the raw word counts for the Bernoulli model (i.e., z_{nw} = min(1, x_{nw} )).
        z = x
        indices = z.nonzero()
        z[indices]=1
        return z


    def fit(self):
        #Find the thresholded MLEs
        self.fitted =1
        if self.method == 1:
            self.mu1 = self.mle(self.zTrain, self.yTrain, 1, self.threshold)
            self.mu0 = self.mle(self.zTrain, 1- self.yTrain, 1, self.threshold)
        if self.method == 0:
            self.thetaInv1 = self.mle(self.xTrain, self.yTrain, 0, self.threshold)
            self.thetaInv0 = self.mle(self.xTrain, 1 - self.yTrain, 0, self.threshold)

    def mle(self, z, y, method, thresh):
        #MLE returns the thresholded MLE
        n = np.sum(y)
        if method == 1:
            x = z.multiply(y)
            mu = np.maximum(x.sum(axis=0)/ n, thresh)
            return mu
        if method == 0:
            x = z.multiply(y)
            thetaInv =  1 + np.maximum(x.sum(axis=0)/n, thresh)
            return thetaInv





    def calculateLL(self, x):
        #calculates the log-likelihood for prediction
        if self.fitted ==0:
            print('Fit first!')
        else:
            if self.method==1:
                logMu1 = np.log(self.mu1.transpose())
                logMu0 = np.log(self.mu0.transpose())
                logMu1m = np.log(1-self.mu1.transpose())
                logMu0m = np.log(1-self.mu0.transpose())
                logL1 = x.dot(logMu1-logMu1m)
                logL1 = logL1 + np.sum(logMu1m)
                logL0 = x.dot(logMu0 - logMu0m)
                logL0 = logL0 + np.sum(logMu0m)
                return [logL0, logL1]
            if self.method==0:
                logTheta0 = 0 - np.log(self.thetaInv0.transpose())
                logTheta1 = 0 - np.log(self.thetaInv1.transpose())
                logTheta0m = np.log(np.maximum(self.thetaInv0.transpose()-1, self.threshold)) - np.log(self.thetaInv0.transpose())
                logTheta1m = np.log(np.maximum(self.thetaInv1.transpose()-1, self.threshold)) - np.log(self.thetaInv1.transpose())
                #logTheta0m = np.log(1-self.theta0.transpose())
                #logTheta1m = np.log(1-self.theta1.transpose())
                logL0 = x.dot(logTheta0m)
                logL0 = logL0 + np.sum(logTheta0)
                logL1 = x.dot(logTheta1m)
                logL1 = logL1 + np.sum(logTheta1)
                return [logL0, logL1]


    def predict(self, x):
        #Predicts whether spam or not-spam.
        [L0, L1] = self.calculateLL(x)
        y = (L1 > L0)
        return (y+0)
    def accuracy(self, y1, y2):
        zz = (y1 ==y2)
        return np.sum(zz)/len(y1)

    def validate(self, thresholds):
        #Accepts a list of thresholds and finds the accuracy of the each classifier on the validation data.
        acc = np.zeros(len(thresholds))
        if self.method==1:
            for i in range(len(thresholds)):
                self.threshold = thresholds[i]
                self.fit()
                acc[i]=self.accuracy(self.yVal,self.predict(self.zVal))
            i = np.argmax(acc)
            self.acc= acc
            self.threshold = thresholds[i]
            self.fit()
        if self.method == 0:
            for i in range(len(thresholds)):
                self.threshold = thresholds[i]
                self.fit()
                acc[i]=self.accuracy(self.yVal,self.predict(self.xVal))
            i = np.argmax(acc)
            self.acc= acc
            self.threshold = thresholds[i]
            self.fit()


            



