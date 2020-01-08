import numpy as np
import problem2_func
from scipy.spatial.distance import cdist


[x_train, y_train, x_test, y_test] = problem2_func.load_digits(200,0,[1, 2, 7])
dist = cdist(x_test,x_train)
xx = np.argpartition(dist, 11, axis=1)
xxx=xx[88, :]
x=dist[88,xxx]


for i in range(11):
    for j in range(11,600):
        if x[i]>x[j]:
            print('whoops: i=', i, ' j=', j)

