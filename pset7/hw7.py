
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_tnc
from scipy.linalg import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt




class twoLayer():
    def __init__(self):
        pass


    def fit(self, X, y, J=5, C=1.0):
        self.J=J
        d = X.shape[1]
        wInit = np.random.uniform(low=-1.0, high = 1.0, size = J+1+(d+1)*J) / X.shape[1]
        self.wInit = wInit
        self.optimizationResult = minimize(objective, wInit, args= (X, y, J, C), method='L-BFGS-B', jac = grad, tol=0.00000001)
        self.w = self.optimizationResult.x
        self.obj = self.optimizationResult.fun


    def probs(self, X):
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        w0, w1 = self.reshape(self.w, self.J, X.shape[1])


        z = logit(Xb.dot(w1.transpose()))

        zb = np.hstack((np.ones((X.shape[0], 1)), z))

        return logit(zb.dot(w0))

    def predict(self, X):
        p = self.probs(X)
        y = - np.ones(p.shape)
        y[np.where(p>0.5)] = 1
        return y

    def acc(self, X, y):
        return np.sum(y == self.predict(X)) / len(y)

    def plot(self, X, y, title='Decision boundaries and training data'):
        if X.shape[1] != 2:
            print('Wrong dimensions')
        else:
            xy = np.mgrid[-1.5:1.5:0.01, -1.5:1.5:0.01].reshape(2, -1).T
            p = self.predict(xy)
            plus = xy[np.where(p > 0)]
            minus = xy[np.where(p<0)]
            s=2
            plt.scatter(plus[:, 0], plus[:, 1], color='#ff9999', s=s, alpha=0.5)
            plt.scatter(minus[:, 0], minus[:, 1], color='#87cefa', s=s, alpha=0.5)

            plusTrain = X[np.where(y >0)]
            minusTrain = X[np.where(y < 0)]
            plt.scatter(plusTrain[:, 0], plusTrain[:, 1], color='red', edgecolors='black')
            plt.scatter(minusTrain[:, 0], minusTrain[:, 1], color='blue',  edgecolors='black')
            plt.title(title)
            plt.show()







    def reshape(self, w, J, d):
        w0 = w[0:J + 1]
        w1 = np.reshape(w[J + 1:], (J, d + 1))
        return w0, w1











def logit(x):
    return 1.0 / ( 1.0 + np.exp(x))

def logL(w, X, y, J):
    w0 = w[0:J+1]
    w1 = np.reshape(w[J+1:],(J, X.shape[1]+1))
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))

    z = logit(Xb.dot(w1.transpose()))

    zb = np.hstack((np.ones((X.shape[0], 1)), z))

    return np.sum(np.log(1+np.exp(y*zb.dot(w0))))

def objective(w, X, y, J, C):
    return C*norm(w) /2 + logL(w, X, y, J)/len(y)

def grad(w, X, y, J, C):
    return C*w + gradLogL(w, X, y, J)/len(y)



def gradLogL(w, X, y, J):
    w0 = w[0:J + 1]
    w1 = np.reshape(w[J + 1:], (J, X.shape[1] + 1))

    Xb = np.hstack((np.ones((X.shape[0], 1)), X))

    z = logit(Xb.dot(w1.transpose()))

    zb = np.hstack((np.ones((X.shape[0], 1)), z))

    grad0 = zb.transpose().dot(y*logit(-y*zb.dot(w0)))

    hold1 = y* logit(- y*(zb.dot(w0)))
    hold1 = (Xb.transpose()*hold1).transpose()
    hold2 = (-logit(Xb.dot(w1.transpose())) * logit(-Xb.dot(w1.transpose())))*w0[1:J+1]
    grad1 = hold2.transpose().dot(hold1)
    grad1 = np.reshape(grad1, J*(X.shape[1]+1))
    return np.concatenate((grad0, grad1))














def optimizer(obj_func, initial_theta, bounds):
    # * 'obj_func' is the objective function to be maximized, which
    #   takes the hyperparameters theta as parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta
    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.

    pg = np.power(10.0, -25)
    eps = np.power(10.0, -10)
    fact = np.power(10.0, 1)


    theta_opt, func_min, convergence_dict = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, pgtol=pg, epsilon = eps, factr=fact)

    return theta_opt, func_min


def optimizer2(obj_func, initial_theta, bounds):
    # * 'obj_func' is the objective function to be maximized, which
    #   takes the hyperparameters theta as parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta
    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.


    theta_opt, func_min, convergence_dict = fmin_tnc(obj_func, initial_theta, bounds=bounds)

    return theta_opt, func_min