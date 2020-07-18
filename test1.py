import numpy as np
import l2distance as l2
import l2DistanceSlow as l2l

X = np.array([[1.0,2.0],[3.0,4.0]])
Z = np.array([[1.0,2.0,3.0],[3.0,4.0,5.0]])


D1 = l2.l2distance(X,Z)
D2 = l2l.l2distanceSlow(X,Z)

a = 11