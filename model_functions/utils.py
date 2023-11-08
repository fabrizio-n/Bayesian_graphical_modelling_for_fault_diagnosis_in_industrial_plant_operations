
import numpy as np
from scipy.linalg import cholesky, cho_solve
from collections import deque
import random

    #################################
    #######       isDAG       #######
    #################################

def isDAG(adj_matrix):
    """
    Checks if a Graph is a DAG (Dyrected Acyclic Graph).

    Args:
    adj_matrix (numpy.ndarray): An Adjacency Matrix.

    Returns:
    Boolean: True if the given Graph is a DAG, False otherwise.
    """
    # Number of vertices in the graph
    num_vertices = len(adj_matrix)
    
    # Count the in-degrees for each vertex
    in_degrees = [0] * num_vertices
    for i in range(num_vertices):
        for j in range(num_vertices):
            in_degrees[i] += adj_matrix[j][i]
    
    # Initialize a queue for topological sorting
    queue = deque()
    
    # Find all vertices with in-degree of 0 and enqueue them
    for i in range(num_vertices):
        if in_degrees[i] == 0:
            queue.append(i)
    
    # Initialize a counter for visited vertices
    visited_count = 0
    
    # Perform topological sorting
    while queue:
        vertex = queue.popleft()
        visited_count += 1
        
        # Reduce the in-degrees of adjacent vertices
        for i in range(num_vertices):
            if adj_matrix[vertex][i] == 1:
                in_degrees[i] -= 1
                if in_degrees[i] == 0:
                    queue.append(i)
    
    # If the visited_count is equal to the number of vertices, it's a DAG
    return visited_count == num_vertices



    #################################
    #######        rDAG       #######
    #################################

def rDAG(q, w):
    """
    Generates a random DAG (Dyrected Acyclic Graph) 
    with q nodes and probability of edge inclusion w.

    Args:
    q: number of nodes.
    w: probability of edge inclusion.

    Returns:
    numpy.ndarray: A Randomly generated Adjacency Matrix for the DAG.
    """
    check_dag = False

    while not check_dag:
        # Initialize DAG matrix
        DAG = np.zeros((q, q))

        # Calculate the number of lower-triangular elements
        num_lower_triangular = q * (q - 1) // 2

        # Generate random values for the lower triangular part of the DAG
        lower_triangular_indices = np.tril_indices(q, k=-1)
        random_values = np.random.binomial(1, w, num_lower_triangular)

        # Fill the lower triangular part of the matrix with random values
        DAG[lower_triangular_indices] = random_values

        check_dag = isDAG(DAG)

    return DAG






    #################################
    #######   is_undirected   #######
    #################################

def is_undirected(adjacency_matrix):
    """
    Checks whether a given Graph is undirected.

    Args:
    adjacency_matrix: The Graph adjacency matrix.

    Returns:
    Boolean: True if the Graph is undirected, False otherwise.
    """
    # Check if the adjacency matrix is symmetric
    return (adjacency_matrix == adjacency_matrix.T).all()






    #################################
    #######      chol2inv     #######
    #################################

def chol2inv(A):
    """
    Compute the inverse of a matrix from its Cholesky decomposition.

    Args:
    A (numpy.ndarray): The original matrix.

    Returns:
    numpy.ndarray: The inverse of the matrix.
    """
    L = cholesky(A, lower=True)  # Compute the lower-triangular Cholesky factor
    A_inv = cho_solve((L, True), np.eye(A.shape[0]))
    
    return A_inv

# Example usage:
# A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
# A_inv = chol2inv(A)
# print("Inverse from Cholesky Decomposition:")
# print(A_inv)

