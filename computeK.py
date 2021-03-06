"""
function K = computeK(kernel_type, X, Z)
computes a matrix K such that Kij=g(x,z);
for three different function linear, rbf or polynomial.

Input:
kernel_type: either 'linear','poly','rbf'
X: n input vectors of dimension d (dxn);
Z: m input vectors of dimension d (dxn);
kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)

OUTPUT:
K : nxm kernel matrix
"""
import numpy as np
from l2distance import l2distance
from l2DistanceSlow import  l2distanceSlow
def computeK(kernel_type, X, Z, kpar):
    assert kernel_type in ['linear', 'poly', 'rbf'], kernel_type + ' is an unrecognized kernel type in computeK'
    
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to computeK'
    
    K = np.zeros((n,m))
    
    if kernel_type == 'linear':
        for i in range(0,n):
            K[i,:] = np.dot(X[:, [i]].T, Z)
    elif kernel_type == 'poly':
        for i in range(0,n):
            K[i,:] = (np.dot(X[:, [i]].T, Z) + 1)**kpar
    elif kernel_type == 'rbf':
        K = np.exp(-kpar*(l2distance(X, Z)**2))
    return K


# a = np.random.normal(0,1,[2,2])
# b = np.random.normal(0,1,[2,3])
#
# K = computeK('rbf',a,b,1)
# print(K)