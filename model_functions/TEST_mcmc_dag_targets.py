import sys
sys.path.append("/Users/fabrizioniro/Library/CloudStorage/GoogleDrive-fabrizion18@gmail.com/My Drive/Tesi/faultdiag-thesis/tesi-python/model")

from utils import rDAG, isDAG
import move_dag as md
import marg_like_dag as mld

import numpy as np
from scipy.stats import beta
from scipy.stats import multinomial
from scipy.stats import dirichlet
from scipy.stats import uniform
from scipy.special import gammaln #logarithm of the abs of the gamma !!!!check previous functions
from scipy.special import loggamma

from progressbar import progressbar

def mcmc_dag_targets(X, S, K, burn, w, a=None, a_k=None, b_k=None, n_all=None):
    # Import necessary libraries here (e.g., numpy, scipy)

    ###########
    ## INPUT ##
    ###########

    # X : (n,q) data matrix

    # S    : number of MCMC iterations
    # burn : burn in period

    # w        : prior probability of edge inclusion in p(D)
    # a_k, b_k : hyper-parameters of the Beta(a_k, b_k) prior on the probability of intervention for any node
    # a        : common shape hyper-parameter for the DAG-Wishart prior

    # n_all : (K,1) vector with group sample sizes (n_1, ..., n_K)

    ############
    ## OUTPUT ##
    ############

    # X : (n,q) input data matrix

    # Graph_post  : (q,q,T) array collecting the T = (S - burn) adjacency matrices of the DAGs visited by the MCMC chain
    # Target_post : (q,K,T) array collecting the T = (S - burn) binary matrices representing the K intervention targets

    #########################
    ## Auxiliary functions ##
    #########################

    import move_dag as md
    import marg_like_dag as mld

    q = X.shape[1]

    if a is None:
        a = q

    Graph_post = np.empty((q, q, S))
    Targets_post = np.zeros((q, K, S))

    num_edges = []

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



    for s in progressbar(range(S)):
        ## Update the graph conditionally on the targets

        # Implement move function here
        Graph_move = md.move(A=Graph)

        Graph_prop = Graph_move['A_new']
        nodes_prop = Graph_move['nodes']
        type_operator = Graph_move['type_operator']

        # Distinguish 3 cases:
        if type_operator == 1:
            # (1) Insert a directed edge
            logprior = np.log(w / (1 - w))
            j_star = nodes_prop[1]

            marg_star = np.sum([mld.marg_like_j(j=j_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])])
            marg = np.sum([mld.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])])
        elif type_operator == 2:
            # (2) Delete a directed edge
            logprior = np.log((1 - w) / w)
            j_star = nodes_prop[1]

            marg_star = np.sum([mld.marg_like_j(j=j_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])])
            marg = np.sum([mld.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])])
        else:
            # (3) Reverse a directed edge
            logprior = 0
            i_star = nodes_prop[0]
            j_star = nodes_prop[1]

            marg_star = (np.sum([mld.marg_like_j(j=i_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, i_star] == k, :], n=np.sum(D[:, i_star] == k), a=a) for k in np.unique(D[:, i_star])]) +
                            np.sum([mld.marg_like_j(j=j_star, dag=Graph_prop, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]))

            marg = (np.sum([mld.marg_like_j(j=i_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, i_star] == k, :], n=np.sum(D[:, i_star] == k), a=a) for k in np.unique(D[:, i_star])]) +
                    np.sum([mld.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, k] == 1, X=X[D[:, j_star] == k, :], n=np.sum(D[:, j_star] == k), a=a) for k in np.unique(D[:, j_star])]))

        # acceptance ratio
        ratio_D = min(0, marg_star - marg + logprior)

        # accept move
        if np.log(np.random.uniform()) < ratio_D:
            Graph = Graph_prop

        Graph_post[:, :, s] = Graph


        # Update the targets given the DAG
        if (s % 5 == 0) and (s > burn):
            # Permutation of intervention indicator for variable j and joint update of the targets
            I_cal_prop = I_cal.copy()
            node_int = np.random.choice(range(q))
            ind_j = abs(I_cal_prop[node_int] - 1)
            I_cal_prop[node_int] = ind_j

            D_prop_list = out_D.copy()

            for k in range(K):
                D_prop_list[k][:, I_cal_prop[:, k] == 1] = k
                D_prop_list[k][:, I_cal_prop[:, k] == 0] = 0

            D_prop = np.vstack(D_prop_list)
            j_star = node_int
            marg_I_prop = np.sum([mld.marg_like_j(j=j_star, dag=Graph, I=I_cal_prop[:, h] == 1, X=X[D_prop[:, j_star] == h, :], n=np.sum(D_prop[:, j_star] == h), a=a) for h in np.unique(D_prop[:, j_star])])
            marg_I = np.sum([mld.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, h] == 1, X=X[D[:, j_star] == h, :], n=np.sum(D[:, j_star] == h), a=a) for h in np.unique(D[:, j_star])])
            log_prior = 0
            ratio = min(0, marg_I_prop - marg_I + logprior)

            # accept move
            if np.log(np.random.uniform()) < ratio:
                I_cal = I_cal_prop
                out_D = D_prop_list

            D = np.vstack(out_D)
            Targets_post[:, :, s] = I_cal
        else:
            I_cal_prop = I_cal.copy()
            target_prop = [None] * K

            for k in np.random.choice(K, K, replace=False):
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
                marg_I_k_prop = np.sum([mld.marg_like_j(j=j_star, dag=Graph, I=I_cal_prop[:, h] == 1, X=X[D_prop[:, j_star] == h, :], n=np.sum(D_prop[:, j_star] == h), a=a) for h in np.unique(D_prop[:, j_star])])
                marg_I_k = np.sum([mld.marg_like_j(j=j_star, dag=Graph, I=I_cal[:, h] == 1, X=X[D[:, j_star] == h, :], n=np.sum(D[:, j_star] == h), a=a) for h in np.unique(D[:, j_star])])
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
            
    return {"X": X, "Graph_post": Graph_post[:, :, (burn + 1):S], "Targets_post": Targets_post[:, :, (burn + 1):S]}
