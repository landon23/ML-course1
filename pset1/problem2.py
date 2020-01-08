import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from scipy.stats import mode
digit_list = [1, 2, 7]
N_samp = 200
N_test = 1000
K=2

mat = scipy.io.loadmat('mnist_all.mat')

mat_l = [mat['train'+str(num)][0:N_samp, :] for num in digit_list]
y_l = [np.full((N_samp, 1), num) for num in digit_list]

if N_test == 0:
    mat_t = [mat['test'+str(num)] for num in digit_list]
    y_t = [np.full((mat['test'+str(num)].shape[0], 1), num) for num in digit_list]
else:
    mat_t = [mat['test'+str(num)][0:N_test, :] for num in digit_list]
    y_t = [np.full((N_test, 1), num) for num in digit_list]

#y_l = [np.full((mat['train'+str(num)].shape[0],1),num) for num in digit_list]

x_train = np.vstack(mat_l)
y_train = np.vstack(y_l)

x_test = np.vstack(mat_t)
y_test = np.vstack(y_t)

dist = cdist(x_test, x_train)

xx = np.argpartition(dist,K,axis=1)
labels = y_train.flatten()[xx[:,0:K]]
label = mode(labels, axis=1)[0]
true = (label==y_test)
print(np.sum(true)/len(true))


def cross_val(matt, digs, N, fold, Krange):
    x_tr = load_samples(matt, digs, N, fold=fold)




def load_samples(matt, digs, N, fold=5):
    return [np.vstack([matt['train'+str(num)][np.random.permutation(matt['train'+str(num)].shape[0]), :][0:N, :]
                       for num in digs]) for _ in range(0, fold)]






#
#
# check = 0
# for num in digit_list:
#     train = 'train'+str(num)
#     hold = mat[train]
#     lenn = hold.shape[0]
#     if check == 0:
#         check =1
#         x_train = hold
#         y_train = np.zeros(lenn)
#         y_train[0:lenn]=num
#     else:
#         x_train = np.concatenate((x_train, hold), axis=0)
#         add = np.zeros(lenn)
#         add[0:lenn]=num
#         y_train = np.concatenate((y_train, add))









