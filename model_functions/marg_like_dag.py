import numpy as np
from scipy.linalg import cholesky
from utils import chol2inv, is_undirected
import utils
from scipy.special import gammaln

# Auxiliary functions

def pa(node, adjacency_matrix):
    # Convert the object (adjacency_matrix) to a NumPy matrix
    amat = np.array(adjacency_matrix)

    # Check if the graph is undirected
    if is_undirected(amat):
        return None

    # Find parents of node j
    parents = np.where(amat[:, node] > 0)[0]
    parents = [p for p in parents if p != node]

    if parents:
        return parents
    else:
        return None

def fa(myset, object):
    return list(myset) + list(pa(myset, object))

# Main function

def marg_like_j(j, dag, I, X, n, a):
    # Compute the marginal likelihood for node j in dag

    g = 1 / n

    q = dag.shape[1]

    dag_int = dag.copy()
    dag_int[:, I] = 0

    pa_j = pa(j, dag_int)
    
    if pa_j == None:
        p_j = 0
    else:
        p_j = len(pa_j)

    y = X[:, j]
    XX = X[:, pa_j]

    j_pos = q - p_j
    a_j = (a + q - 2 * j_pos + 3) / 2 - p_j / 2 - 1

    if not pa_j:
        m = 0.5*a_j*np.log(0.5*g) - 0.5*(a_j+n)*np.log(0.5*g + np.sum(np.square(y))/2) + \
            gammaln(0.5*(a_j+n)) - gammaln(0.5*a_j) - 0.5*n*np.log(2*np.pi)
    else:
        XX = np.asmatrix(XX)

        # Compute the inverse of the Cholesky factor
        L_inv = chol2inv(np.diag([g]*p_j) + XX.T @ XX)
        # Compute U_jj
        U_jj = (y.T @ XX) @ L_inv @ (XX.T @ y).T

        m = -0.5*n*np.log(2*np.pi) + 0.5*np.log(g**p_j) - \
                0.5*np.log(np.linalg.det(np.identity(p_j)*g+XX.T@XX)) + \
                0.5*a_j*np.log(0.5*g) - 0.5*(a_j+n)*np.log(g/2+np.sum(np.square(y))/2 - U_jj/2) + \
                gammaln(0.5*(a_j+n)) - gammaln(0.5*a_j)
        

    return float(np.matrix(m)[0,0])



