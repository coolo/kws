#! /usr/bin/python3

import numpy as np
from numpy.core.numeric import indices
from sklearn.cluster import KMeans

incides = []
for i in range(ord('a'), ord('z')+1):
    incides.append(chr(i))
for i in range(ord('A'), ord('Z')+1):
    incides.append(chr(i))

def difference_between(vec1, vec2):
    return np.linalg.norm(vec1-vec2)

max=0
best_hit=None

def array_to_str(array):
    str=""
    for x in array:
        str += chr(int(x * 10 + 0.5) + ord('0'))
    return str

over=np.zeros((1, 36), dtype=np.float32)

with open('labels_in.txt', 'r') as f:
    for line in f:
        array=[ord(i)-ord('0') for i in line.strip()]
        array = np.array(array, dtype=np.float32) / 10
        over = np.vstack((over, array))

kmeans = KMeans(n_clusters=len(incides), random_state=0, verbose=1).fit(over)
centers = kmeans.cluster_centers_
for i, k in enumerate(centers):
    line = array_to_str(k)
    print(f"'{incides[i]}': np.array([" + ",".join([i for i in line.strip()]) + "], dtype=np.float32) / 10,")
 