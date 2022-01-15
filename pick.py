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
# map to -1 .. -1
over = np.clip(over + 10, -10, 10) / 10
import matplotlib.pyplot as plt
mu = 10
# map to 0..1
over = (np.sign(over) * np.log(1.0 + mu * np.abs(over)) / np.log(1.0 + mu) + 1) / 2

kmeans = KMeans(n_clusters=len(incides), random_state=0, verbose=1, max_iter=1).fit(over)
centers = np.int8(kmeans.cluster_centers_ * 10)
centers = np.unique(centers, axis=0)
for i, k in enumerate(centers):
    line = array_to_str(k)
    print(f"'{incides[i]}': np.array([" + ",".join([i for i in line.strip()]) + "], dtype=np.float32) / 10,")
 
