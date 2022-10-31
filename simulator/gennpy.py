import numpy as np
import sys

size = sys.argv[2]
filename = sys.argv[1]

inds = open(filename+'.ind', 'r')
kvalues = open(filename+'.kval', 'r')

k1 = np.zeros((size, size))
k2 = np.zeros((size, size))
k3 = np.zeros((size, size))

for ind, kval in zip(inds, kvalues):
    i = list(map(int, ind.split()))[0]
    j = list(map(int, ind.split()))[1]
    k1[i, j] = list(map(float, kval.split()))[0]
    k2[i, j] = list(map(float, kval.split()))[1]
    k3[i, j] = list(map(float, kval.split()))[2]

inds.close()
kvalues.close()

np.save('k1', k1)
np.save('k2', k2)
np.save('k3', k3)
