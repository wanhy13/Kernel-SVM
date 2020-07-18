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

def l2distance(X,Z):

    d, n = X.shape
    d, m = Z.shape
    # test = np.power(X,2)
    # k1 = np.sum(np.square(X), axis=0).reshape(-1, 1)
    # k2 = np.sum(Z**2, axis=0).reshape(1,-1)
    # c = np.repeat(np.sum(Z**2, axis=0).reshape(1,-1), (n,1)).T
    # S = np.tile(np.sum(np.square(X), axis=0).reshape(-1, 1), m)
    # G = X.T.dot(Z)
    # gua = np.sum(np.square(Z), axis=0).reshape(1,-1).T
    # R = np.repeat(np.sum(np.square(Z), axis=0).reshape(1,-1).T,n,axis=1).T
    D = np.tile(np.sum(np.square(X), axis=0).reshape(-1, 1), m) - 2 * X.T.dot(Z) + np.repeat(np.sum(np.square(Z), axis=0).reshape(1,-1).T,n,axis=1).T
    D[D < 0] = 0
    D = np.power(D, 0.5)
    # assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'

    # D = np.zeros((n, m))
    #
    # for i in range(0, n):
    #     Xnew = np.tile(X[:, [i]], (1, m))
    #     D[i, :] = np.sqrt(np.sum((Xnew - Z) ** 2, 0))
    # D[D<0] = 0
    return D

# a = np.random.normal(0,1,[100,2])
# b = np.random.normal(0,1,[100,3])
# # v = (a-b)**2
# # s1 = np.sqrt(np.sum(v,0))
# s2 = l2distance(a,b)
# print(s2)