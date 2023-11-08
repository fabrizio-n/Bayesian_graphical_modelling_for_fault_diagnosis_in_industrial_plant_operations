import numpy as np
#from scipy.linalg import cholesky, cholesky_inverse
from numpy.random import multivariate_normal

# Function to generate interventional data for a single intervention
def gen_int_data(A_true, B, Sigma, I_k, sigma_I, n_k, k):
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
def gen_dataset(A_true, B, Sigma, I_cal, sigma_int, n_all):
    K = len(n_all)
    out_X = [gen_int_data(A_true, B, Sigma, I_k=np.where(I_cal[:, k] == 1)[0],
                         sigma_I=sigma_int[k], n_k=n_all[k], k=k) for k in range(K)]
    out_D = [np.zeros((n_all[k], A_true.shape[0])) for k in range(K)]

    for k in range(K):
        out_D[k][np.where(I_cal[:, k] == 1)[0], :] = k + 1

    return {"X": np.vstack(out_X), "D": np.vstack(out_D)}
