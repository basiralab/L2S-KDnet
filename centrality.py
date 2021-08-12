import networkx as nx
import numpy as np


# This file contains 2 functions: The first one (to_2d) create a similarity matrix from a vector
# the second one uses the to_2d function to generate a matrix and then compute its centrality vector.

# put it back into a 2D symmetric array
def to_2d(vector, size):
    x = np.zeros((size, size))
    c = 0
    for i in range(1, size):
        for j in range(0, i):
            x[i][j] = vector[c]
            x[j][i] = vector[c]
            c = c + 1
    return x

#function that calculates eigenvector centrality and pagerank
def topological_measures(data, size, eigenvector=False):
    EC = np.empty((0, size), int)
    PC = np.empty((0, size), int)
    topology = []
    max_solver_iterations = 1000000000
    for i in range(data.shape[0]):
        A = to_2d(data[i], size)
        np.fill_diagonal(A, 0)

        # create a graph from similarity matrix
        G = nx.from_numpy_matrix(A)
        U = G.to_undirected()

        # # compute eigenvector centrality and transform the output to vector
        ec = nx.eigenvector_centrality(U, weight="weight", max_iter=max_solver_iterations)
        eigenvector_centrality = np.array([ec[g] for g in U])
        EC = np.vstack((EC, eigenvector_centrality))

        if not eigenvector:
            # compute pagerank
            pr = nx.pagerank(U, alpha=0.85, weight="weight", max_iter=max_solver_iterations)
            pagerank = np.array([pr[g] for g in U])
            PC = np.vstack((PC, pagerank))

    topology.append(EC)
    if not eigenvector:
        topology.append(PC)

    return topology
