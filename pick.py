#! /usr/bin/python3

import numpy as np
from numpy.core.numeric import indices
from sklearn.cluster import KMeans
import os

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
        str += chr(x + ord('0'))
    return str

over=np.zeros((1, 36), dtype=np.float32)
data = np.load('all-waves.npz', mmap_mode='r')['x']
length=data[0].shape[0]
over=np.zeros((length * len(list(data)), 36), dtype=np.float32)
for i, x in enumerate(data):
    over[i*length:(i+1)*length,0:36] = x

# the majority is at around -7, so move that to be 0
over += 7

# map to -1 .. -1 (from -10..10)
over = np.clip(over, -10, 10) / 10

mu = 10
# map to 0..1 using mu-law
over = (np.sign(over) * np.log(1.0 + mu * np.abs(over)) / np.log(1.0 + mu) + 1) / 2

kmeans = KMeans(n_clusters=len(incides), verbose=1, n_init=2).fit(over)
centers = np.int8(kmeans.cluster_centers_ * 10)
centers = np.unique(centers, axis=0)

touched=True
# good old bubble sort
while touched:
    touched=False
    for i in range(1, len(centers)):
        if np.mean(centers[i-1]) > np.mean(centers[i]):
            centers[[i,i-1]] = centers[[i-1,i]]
            touched=True
            break
for i, k in enumerate(centers):
    line = array_to_str(k)
    print(f"'{incides[i]}': np.array([" + ",".join([i for i in line.strip()]) + "], dtype=np.float32) / 10,")
 
