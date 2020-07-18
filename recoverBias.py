"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    dist = C/2
    index = 0
    for ind in range(len(alphas)):
        newdist = abs(alphas[ind]-C/2)
        if newdist < dist:
            dist = newdist
            index = ind
    bias = yTr[index] - np.dot(alphas.T*yTr.T, K[:, index])

    return bias
