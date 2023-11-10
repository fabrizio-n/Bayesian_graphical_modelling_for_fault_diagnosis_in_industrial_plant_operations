
import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import cholesky, cho_solve
from scipy.special import gammaln
from collections import deque
from tqdm.notebook import tqdm_notebook



class MCMCDagTargets:
    def __init__(self, seed = None):

        self.seed = seed


    
    def gen_int_data(self, A_true, B, Sigma, I_k, sigma_I, n_k, k):
        """
        Checks if a Graph is a DAG (Dyrected Acyclic Graph).

        Args:
        adj_matrix (numpy.ndarray): An Adjacency Matrix.

        Returns:
        Boolean: True if the given Graph is a DAG, False otherwise.
        """
        
        q = A_true.shape[0]
        A_int = A_true.copy()
        A_int[:, I_k] = 0  # Remove all edges "pointing" to the intervened node (all j --> I)

        B_I = np.multiply(A_int, B)  # Remove the regression coefficients associated with j -> I

        Sigma_I = Sigma.copy()

        for h in I_k:
            Sigma_I[h, h] = sigma_I

        np.fill_diagonal(B_I, 1)

        S_I = np.dot(np.dot(np.linalg.inv(B_I.T), Sigma_I), np.linalg.inv(B_I))

        mean = np.zeros(q)
        X = multivariate_normal(mean=mean, cov=S_I, size=n_k)

        m = np.mean(X, axis=0)
        X = X - m  # Center the data by subtracting the column-wise mean

        return X

    # Main function to generate the entire dataset
    def gen_dataset(self, A_true, B, Sigma, I_cal, sigma_int, n_all):
        """
        Checks if a Graph is a DAG (Dyrected Acyclic Graph).

        Args:
        adj_matrix (numpy.ndarray): An Adjacency Matrix.

        Returns:
        Boolean: True if the given Graph is a DAG, False otherwise.
        """
        K = len(n_all)
        out_X = [self.gen_int_data(A_true, B, Sigma, I_k=np.where(I_cal[:, k] == 1)[0],
                             sigma_I=sigma_int[k], n_k=n_all[k], k=k) for k in range(K)]
        out_D = [np.zeros((n_all[k], A_true.shape[0])) for k in range(K)]

        for k in range(K):
            out_D[k][np.where(I_cal[:, k] == 1)[0], :] = k + 1

        return {"X": np.matrix(np.vstack(out_X)), "D": np.matrix(np.vstack(out_D))}


    def isDAG(self, adj_matrix):
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
    

    def is_undirected(self, adjacency_matrix):
        """
        Checks whether a given Graph is undirected.

        Args:
        adjacency_matrix: The Graph adjacency matrix.

        Returns:
        Boolean: True if the Graph is undirected, False otherwise.
        """
        # Check if the adjacency matrix is symmetric
        return (adjacency_matrix == adjacency_matrix.T).all()


    def rDAG(self, q, w, seed = None):
        """
        Generates a random DAG (Dyrected Acyclic Graph) 
        with q nodes and probability of edge inclusion w.

        Args:
        q: number of nodes.
        w: probability of edge inclusion.

        Returns:
        numpy.ndarray: A Randomly generated Adjacency Matrix for the DAG.
        """
        if self.seed:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
                     

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

            check_dag = self.isDAG(DAG)

        return DAG
    
    
    def shd(self, matrix_A, matrix_B):
        """
        Compute the structural hamming distance between two matrices.

        Args:
        A (numpy.ndarray)
        B (numpy.ndarray)

        Returns:
        int: The structural hamming distance of matrix A from matrix B.
        """        
        # Check for consistency of matrix dimensions
        if matrix_A.shape != matrix_B.shape:
            raise ValueError("Adjacency matrices have different dimensions.")

        # Compute the structural Hamming distance
        diff_matrix = np.abs(matrix_A - matrix_B)
        hamming_distance = np.sum(diff_matrix)

        return hamming_distance
    

    def chol2inv(self, A):
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
    
    
    def count_edges(self, A):
        """
        Count the number of edges in a Directed Acyclic Graph (DAG) represented as an adjacency matrix.

        Parameters:
        - adjacency_matrix (np.ndarray): The adjacency matrix of the DAG.

        Returns:
        - int: The number of edges in the DAG.
        """
        # Ensure the matrix is square and binary
        if not isinstance(A, np.ndarray) or A.shape[0] != A.shape[1]:
            raise ValueError("Input must be a square numpy array representing a binary adjacency matrix.")

        # Count the number of edges in the upper triangle of the matrix
        num_edges = np.sum(A != 0)

        return num_edges
    

    def id(self, A, nodes):
        """
        Inserts an edge in the adjacency matrix of the DAG.

        Args:
        A (numpy.ndarray): An adjacency matrix.

        Returns:
        Graph (numpy.ndarray): The modified adjacency matrix.
        """
        Graph = A.copy()
        x = nodes[0]
        y = nodes[1]
        Graph[x, y] = 1
        return Graph


    def dd(self, A, nodes):
        """
        Deletes an edge in the adjacency matrix of the DAG.

        Args:
        A (numpy.ndarray): An adjacency matrix.

        Returns:
        Graph (numpy.ndarray): The modified adjacency matrix.
        """
        Graph = A.copy()
        x = nodes[0]
        y = nodes[1]
        Graph[x, y] = 0
        return Graph


    def rd(self, A, nodes):
        """
        Reverts an edge in the adjacency matrix of the DAG.

        Args:
        A (numpy.ndarray): An adjacency matrix.

        Returns:
        Graph (numpy.ndarray): The modified adjacency matrix.
        """
        Graph = A.copy()
        x = nodes[0]
        y = nodes[1]
        Graph[x, y] = 0
        Graph[y, x] = 1
        return Graph


    def move(self, A):
        """
        Perform a random local move from the input DAG (with adjacency matrix A) to an adjacent DAG.

        Args:
        A (numpy.ndarray): An adjacency matrix.

        Returns:
        dict: A_new, type_operator, nodes.
        """
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
                A_succ = self.id(A = A, nodes = [O[i, 1], O[i, 2]])
            elif O[i, 0] == 1:
                A_succ = self.dd(A = A, nodes = [O[i, 1], O[i, 2]])
            else:
                A_succ = self.rd(A = A, nodes = [O[i, 1], O[i, 2]])

            val = self.isDAG(A_succ)

            if val:
                break

        A_new = A_succ #.copy()

        return {"A_new": A_new, "type_operator": O[i, 0], "nodes": O[i, 1:3]}
    

    def pa(self, node, adjacency_matrix):
        """
        Finds the coordinates of the parents 
        for a given node in a given adjacency matrix.

        Args:
        A (numpy.ndarray): An adjacency matrix.

        Returns:
        parents (list): The coordinates of the parents 
        for a given node in a given adjacency matrix.
        """
        # Convert the object (adjacency_matrix) to a NumPy matrix
        amat = np.array(adjacency_matrix)

        # Check if the graph is undirected
        if self.is_undirected(amat):
            return None

        # Find parents of node j
        parents = np.where(amat[:, node] > 0)[0]
        parents = [p for p in parents if p != node]

        if parents:
            return parents
        else:
            return None

    def fa(self, myset, object):
        """
        Finds the coordinates of the parents and childs 
        for a given node in a given adjacency matrix.

        Args:
        A (numpy.ndarray): An adjacency matrix.

        Returns:
        parents (list): The coordinates of the parents and childs 
        for a given node in a given adjacency matrix.
        """
        return list(myset) + list(pa(myset, object))

    # Main function

    def marg_like_j(self, j, dag, I, X, n, a):
        """
        Computes the marginal likelihood for a given node j.

        Args:
        j, dag, I, X, n, a.

        Returns:
        m (float): The marginal likelihood for a given node j.
        """
        # Compute the marginal likelihood for node j in dag

        g = 1 / n

        q = dag.shape[1]

        dag_int = dag.copy()
        dag_int[:, I] = 0

        pa_j = self.pa(j, dag_int)

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
            L_inv = self.chol2inv(np.diag([g]*p_j) + XX.T @ XX)
            # Compute U_jj
            U_jj = (y.T @ XX) @ L_inv @ (XX.T @ y) ########

            m = -0.5*n*np.log(2*np.pi) + 0.5*np.log(g**p_j) - \
                    0.5*np.log(np.linalg.det(np.identity(p_j)*g+XX.T@XX)) + \
                    0.5*a_j*np.log(0.5*g) - 0.5*(a_j+n)*np.log(g/2+np.sum(np.square(y))/2 - U_jj/2) + \
                    gammaln(0.5*(a_j+n)) - gammaln(0.5*a_j)


        return float(np.matrix(m)[0,0])
    

    def mcmc_dag_targets(self, X, S, burn, w, a=None, a_k=None, b_k=None, n_all=None, seed=None, hide_pbar=False):
        """
        Main MCMC Algorithm.

        Args:
        S        : number of MCMC iterations
        burn     : burn in period
        w        : prior probability of edge inclusion in p(D)
        a_k, b_k : hyper-parameters of the Beta(a_k, b_k) prior on the probability of intervention for any node
        a        : common shape hyper-parameter for the DAG-Wishart prior
        n_all    : (K,1) vector with group sample sizes (n_1, ..., n_K)

        Returns:
        X           : (n,q) input data matrix
        Graph_post  : (q,q,T) array collecting the T = (S - burn) adjacency matrices of the DAGs visited by the MCMC chain
        Target_post : (q,K,T) array collecting the T = (S - burn) binary matrices representing the K intervention targets
        """

        if self.seed:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
                
        q = X.shape[1]
        K = len(n_all)

        if a is None:
            a = q

        Graph_post = np.empty((q, q, S))
        trace_edges = np.empty(S)
        Targets_post = np.zeros((q, K, S))

        # Initialize the chain
        Graph = np.zeros((q, q))
        I_cal = np.zeros((q, K))

        D = [np.zeros((n_all[k], q), dtype=int) for k in range(K)]

        for k in range(K):
            D[k][:, I_cal[:, k] == 1] = k

        D = np.vstack(D)
        Graph_post[:, :, 0] = Graph
        out_D = [np.zeros((n_all[k], q), dtype=int) for k in range(K)]

        for k in range(K):
            out_D[k][:, I_cal[:, k] == 1] = k

        for s in tqdm_notebook(range(S), desc = 'MCMC Sampling', disable = hide_pbar):
            ## Update the graph conditionally on the targets

            Graph_move = self.move(A=Graph)

            Graph_prop = Graph_move['A_new']
            nodes_prop = Graph_move['nodes']
            type_operator = Graph_move['type_operator']

            if type_operator == 0:
                # (0) Insert a directed edge
                logprior = np.log(w / (1 - w))
                j_star = nodes_prop[1]

                marg_star = np.sum(
                    [self.marg_like_j(j=j_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]
                    )
                marg = np.sum(
                    [self.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]
                    )
            
            elif type_operator == 1:
                # (1) Delete a directed edge
                logprior = np.log((1 - w) / w)
                j_star = nodes_prop[1]

                marg_star = np.sum(
                    [self.marg_like_j(j=j_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]
                    )
                marg = np.sum(
                    [self.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]
                    )
            
            else:
                # (2) Reverse a directed edge
                logprior = 0
                i_star = nodes_prop[0]
                j_star = nodes_prop[1]

                marg_star = (
                    np.sum(
                    [self.marg_like_j(j=i_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, i_star] == k, :], n=np.sum(D[:, i_star] == k), a=a) for k in np.unique(D[:, i_star])]
                    ) +
                                np.sum(
                                    [self.marg_like_j(j=j_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]
                                    )
                                )
                marg = (
                    np.sum(
                        [self.marg_like_j(j=i_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, i_star] == k, :], n=np.sum(D[:, i_star] == k), a=a) for k in np.unique(D[:, i_star])]) +
                        np.sum(
                            [self.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]
                            )
                        )

            # acceptance ratio
            ratio_D = min(0, marg_star - marg + logprior)

            # accept move
            if np.log(np.random.uniform()) < ratio_D:
                Graph = Graph_prop#.copy()

            Graph_post[:, :, s] = Graph
            trace_edges[s] = self.count_edges(Graph)


            # Update the targets given the DAG
            if (s % 5 == 0) and (s > burn):
                # Permutation of intervention indicator for variable j and joint update of the targets
                I_cal_prop = I_cal.copy()
                node_int = np.random.choice(range(q))
                ind_j = abs(I_cal_prop[node_int, 1:] - 1)
                I_cal_prop[node_int, 1:] = ind_j
                D_prop_list = out_D.copy()

                for k in range(1, K):
                    D_prop_list[k][:, I_cal_prop[:, k] == 1] = k
                    D_prop_list[k][:, I_cal_prop[:, k] == 0] = 0

                D_prop = np.vstack(D_prop_list)
                j_star = node_int#.copy()
                marg_I_prop = np.sum(
                    [self.marg_like_j(j=j_star, dag=Graph, I=I_cal_prop[:, h] == 1, X=X[D_prop[:, j_star] == h, :], n=np.sum(D_prop[:, j_star] == h), a=a) for h in np.unique(D_prop[:, j_star])]
                    )
                marg_I = np.sum(
                    [self.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, h] == 1, X=X[D[:, j_star] == h, :], n=np.sum(D[:, j_star] == h), a=a) for h in np.unique(D[:, j_star])]
                    )
                ratio = min(0, marg_I_prop - marg_I + logprior)

                # accept move
                if np.log(np.random.uniform()) < ratio:
                    I_cal = I_cal_prop#.copy()
                    out_D = D_prop_list#.copy()

                D = np.vstack(out_D)
                Targets_post[:, :, s] = I_cal
            else:
                I_cal_prop = I_cal.copy() ####################
                target_prop = [None] * K

                for k in np.random.choice(range(1,K), K-1, replace=False):
                    target_prop[k] = np.random.choice(range(q), 1)[0]

                    if I_cal[target_prop[k], k] == 0:
                        I_cal_prop[target_prop[k], k] = 1
                    else:
                        I_cal_prop[target_prop[k], k] = 0

                    D_k_prop = np.zeros((n_all[k], q), dtype=int)
                    D_k_prop[:, I_cal_prop[:, k] == 1] = k
                    out_D_tmp = out_D.copy()
                    out_D_tmp[k] = D_k_prop
                    D_prop = np.vstack(out_D_tmp)
                    j_star = target_prop[k]
                    marg_I_k_prop = np.sum([self.marg_like_j(j=j_star, dag=Graph, I=I_cal_prop[:, h] == 1, X=X[D_prop[:, j_star] == h, :], n=np.sum(D_prop[:, j_star] == h), a=a) for h in np.unique(D_prop[:, j_star])])
                    marg_I_k = np.sum([self.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, h] == 1, X=X[D[:, j_star] == h, :], n=np.sum(D[:, j_star] == h), a=a) for h in np.unique(D[:, j_star])])
                    logprior_k = (gammaln(a_k + np.sum(I_cal_prop[:, k])) + gammaln(q - np.sum(I_cal_prop[:, k]) + b_k) -
                                    gammaln(a_k + np.sum(I_cal[:, k])) - gammaln(q - np.sum(I_cal[:, k]) + b_k))

                    ratio_k = min(0, marg_I_k_prop - marg_I_k + logprior_k)

                    # accept move
                    if np.log(np.random.uniform()) < ratio_k:
                        I_cal[:, k] = I_cal_prop[:, k]

                    out_D[k] = np.zeros((n_all[k], q), dtype=int)
                    out_D[k][:, I_cal[:, k] == 1] = k
                    D = np.vstack(out_D)

                Targets_post[:, :, s] = I_cal
        
        I_cal_hat = np.round(np.mean(Targets_post, axis=2), 2)
        P_hat = np.round(np.mean(Graph_post, axis=2), 2)
        A_hat = np.round(P_hat)


        return {
            #"X": X,
            "Targets_estimate": I_cal_hat,
            "P_DAG_estimate": P_hat,
            "DAG_estimate": A_hat,
            "Trace_DAG_edges":trace_edges
            }



