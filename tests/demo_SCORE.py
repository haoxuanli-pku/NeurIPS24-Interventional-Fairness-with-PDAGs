from stein import *
import numpy as np
import cdt

def generate_right(d, s0, N, noise_std = 1.5, noise_type = 'Gauss', graph_type = 'ER', GP = True, lengthscale=1):
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
#     print(adjacency)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP = GP, lengthscale=lengthscale)
    X, noise_var, noise, beta, Y, node = teacher.sample_right(N)
    return X, adjacency, noise, beta, Y, node
