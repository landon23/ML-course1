import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from scipy.stats import mode
import math


def load_digits(N_samp,N_test,digit_list):
    #This function loads grey-scale images of handwritten digits from the mnist_all.mat file.  It outputs train and
    # test data matrices x_train and x_test of size Samples x Features.
    #If N_test = 0, then it returns all of the test data.

    mat = scipy.io.loadmat('mnist_all.mat')

    mat_l = [mat['train' + str(num)][0:N_samp, :] for num in digit_list]
    y_l = [np.full((N_samp, 1), num) for num in digit_list]

    if N_test == 0:
        mat_t = [mat['test' + str(num)] for num in digit_list]
        y_t = [np.full((mat['test' + str(num)].shape[0], 1), num) for num in digit_list]
    else:
        mat_t = [mat['test' + str(num)][0:N_test, :] for num in digit_list]
        y_t = [np.full((N_test, 1), num) for num in digit_list]

    x_train = np.vstack(mat_l)
    y_train = np.vstack(y_l)

    x_test = np.vstack(mat_t)
    y_test = np.vstack(y_t)

    return [x_train, y_train, x_test, y_test]

def classify(x_train, y_train, x, K):
    # takes training data x_train, y_train and classifies test data x using K-NN.

    #Finds pairwise distance matrix, then the K closest training examples to each data point in x and outputs the most common label.
    dist = cdist(x, x_train)
    xx = np.argpartition(dist, K, axis=1)
    labels = y_train.flatten()[xx[:, 0:K]]
    label = mode(labels, axis=1)[0]
    return label


def accuracy(y_test,y):
    true = (y == y_test)
    return (np.sum(true) / len(true))

def cross_valid_acc(x,y,folds,K):
    per = np.random.permutation(x.shape[0])
    block = math.floor(x.shape[0]/folds)
    acc = np.zeros(folds)
    for i in range(folds):
        if i == folds-1:
            cur = per[i*block:x.shape[0]]
        else:
            cur = per[i*block:(i+1)*block]
        x_train = x[cur,:]
        y_train = y[cur,:]
        x_test = np.delete(x,cur,axis=0)
        y_test = np.delete(y,cur,axis=0)
        yy = classify(x_train, y_train, x_test, K)
        acc[i] = accuracy(y_test,yy)

    return np.mean(acc)






