import numpy as np
import random
from utils import isDAG

# Auxiliary functions

def n_edge(A):
    return len(np.where((A[np.tril_indices(A.shape[0])] == 1) | (np.transpose(A)[np.tril_indices(A.shape[0])] == 1)))

names = ["action", "test", "x", "y"]
actions = ["id", "dd", "rd"]

# 3 types of possible moves indexed by {1,2,3}

# 1: insert a directed edge (id)
# 2: delete a directed edge (dd)
# 3: reverse a directed edge (rd)

# A: adjacency matrix of the input DAG
# nodes: nodes (x, y) involved in the action

def id(A, nodes):
    Graph = A.copy()
    x = nodes[0]
    y = nodes[1]
    Graph[x, y] = 1
    return Graph

def dd(A, nodes):
    Graph = A.copy()
    x = nodes[0]
    y = nodes[1]
    Graph[x, y] = 0
    return Graph

def rd(A, nodes):
    Graph = A.copy()
    x = nodes[0]
    y = nodes[1]
    Graph[x, y] = 0
    Graph[y, x] = 1
    return Graph

# Main function

def move(A):
    # Perform a random local move from the input DAG (with adjacency matrix A) to an adjacent DAG
    # A: adjacency matrix of the input DAG
    # q: number of vertices

    A_na = A.copy()
    np.fill_diagonal(A_na, np.nan)

    id_set = np.array([], dtype=int)
    dd_set = np.array([], dtype=int)
    rd_set = np.array([], dtype=int)

    # Set of all possible operators
    sets = []

    # Set of possible nodes for id
    set_id = np.argwhere(A_na == 0) 

    if len(set_id) != 0:
        id_set = np.column_stack((np.zeros(len(set_id), dtype=int), set_id))
        sets.append(id_set)

    # Set of possible nodes for dd
    set_dd = np.argwhere(A_na == 1)

    if len(set_dd) != 0:
        dd_set = np.column_stack((np.ones(len(set_dd), dtype=int), set_dd))
        sets.append(dd_set)

    # Set of possible nodes for rd
    set_rd = np.argwhere(A_na == 1)

    if len(set_rd) != 0:
        rd_set = np.column_stack((2 * np.ones(len(set_rd), dtype=int), set_rd))
        sets.append(rd_set)


    O = np.vstack(sets)

    # Sample one of the possible graphs, each obtained by applying an operator in O
    # Check that the proposed graph is a DAG
    while True:
        i = np.random.choice(range(O.shape[0]))

        if O[i, 0] == 0:
            A_succ = id(A = A, nodes = [O[i, 1], O[i, 2]])
        elif O[i, 0] == 1:
            A_succ = dd(A = A, nodes = [O[i, 1], O[i, 2]])
        else:
            A_succ = rd(A = A, nodes = [O[i, 1], O[i, 2]])

        val = isDAG(A_succ)

        if val:
            break

    A_new = A_succ

    return {"A_new": A_new, "type_operator": O[i, 0], "nodes": O[i, 1:3]}

