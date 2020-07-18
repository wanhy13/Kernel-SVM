import numpy as np

"""
function D=l2distance(X,Z)

Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""


def l2distanceSlow(X, Z):
    d, n = X.shape
    dd, m = Z.shape

    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    D = np.zeros((n, m))

    for i in range(0, n):
        Xnew = np.tile(X[:, [i]], (1, m))
        D[i, :] = np.sqrt(np.sum((Xnew - Z) ** 2, 0))
    D[D<0] = 0
    return D

# a = np.random.normal(0,1,[100,2])
# b = np.random.normal(0,1,[100,3])
# # v = (a-b)**2
# # s1 = np.sqrt(np.sum(v,0))
# s2 = l2distance(a,b)
# print(s2)