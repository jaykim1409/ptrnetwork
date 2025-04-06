import numpy as np 
import pandas as pd 
import random
import math
import matplotlib.pyplot as plt 
from collections import defaultdict
from itertools import combinations
import pickle

import gurobipy as gp
from gurobipy import GRB

from tsp import *


DATA_LEN = 4096
npoints = 20

def rearrange_tour(tour):
    idx = tour.index(0)  # 0의 위치를 찾음
    return tour[idx:] + tour[:idx]  # 0부터 뒤쪽, 0 이전까지 앞으로 붙임


X = defaultdict()
dist_info = defaultdict()
y = defaultdict()



for i in range(DATA_LEN):
    random.seed(i)
    
    nodes = list(range(npoints))
    
    points = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for i in nodes]
    
    distances = {
        (i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
        for i, j in combinations(nodes, 2)
    }

    tour, _ = solve_tsp(nodes, distances)
    
    X[i] = points
    dist_info[i] = distances
    y[i] = rearrange_tour(tour)
    
    
    
# save data
with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('dist_info.pkl', 'wb') as f:
    pickle.dump(dist_info, f)
with open('y.pkl', 'wb') as f:
    pickle.dump(y, f)