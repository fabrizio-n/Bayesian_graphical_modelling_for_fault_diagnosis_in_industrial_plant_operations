from mcmcdagtargets import MCMCDagTargets
import numpy as np 
from IPython.display import clear_output

mcmc = MCMCDagTargets()


def fpr(actual, predicted):
    """
    Compute False Positive Rate (FPR) for two matrices.

    Parameters:
    - actual: 2D array, actual labels (0 or 1)
    - predicted: 2D array, predicted labels (0 or 1)

    Returns:
    - FPR: float, False Positive Rate
    """
    # Flatten matrices to 1D arrays
    actual_flat = np.ravel(actual)
    predicted_flat = np.ravel(predicted)

    # Calculate False Positives (FP)
    false_positives = np.sum((actual_flat == 0) & (predicted_flat == 1))

    # Calculate True Negatives (TN)
    true_negatives = np.sum((actual_flat == 0) & (predicted_flat == 0))

    # Calculate False Positive Rate (FPR)
    fpr = false_positives / (false_positives + true_negatives)

    return fpr


def fnr(actual, predicted):
    """
    Compute False Negative Rate (FNR) for two matrices.

    Parameters:
    - actual: 2D array, actual labels (0 or 1)
    - predicted: 2D array, predicted labels (0 or 1)

    Returns:
    - FNR: float, False Negative Rate
    """
    # Flatten matrices to 1D arrays
    actual_flat = np.ravel(actual)
    predicted_flat = np.ravel(predicted)

    # Calculate False Negatives (FN)
    false_negatives = np.sum((actual_flat == 1) & (predicted_flat == 0))

    # Calculate True Positives (TP)
    true_positives = np.sum((actual_flat == 1) & (predicted_flat == 1))

    # Calculate False Negative Rate (FNR)
    fnr = false_negatives / (false_negatives + true_positives)

    return fnr


def merge_sim_res(inp_dict:dict):
    """
    Merge simulation results dictionary into single output dictionary

    Parameters:
    - inp_dict: dict

    Returns:
    - out_dict: dict
    """
    out_dict = {}
    for key, value in inp_dict.items():
        if key in out_dict:
            out_dict[key].extend(value)
        else:
            out_dict[key] = value
    return out_dict


def simulate(sim_runs):
    """
    Run simulations for a fixed DAG and different n_k dimensions.
    Designed to run with python multiprocessing.
    
    Args:
    asim_runs (list): list of ranges to iterate in parallel for simulation.
    
    Returns:
    Boolean: True if the given Graph is a DAG, False otherwise.
    """    
    S = 50000  # Fixed parameter 1
    burn = 20000  # Fixed parameter 2
    A = mcmc.rDAG(q = 20, w = 2/20, seed = 1998)
    K = 2
    sim_n_k = [20, 100, 500, 2000]    

    q = A.shape[1]
    a_k = 1 / q
    b_k = 1
    w = 0.1

    out_sim_i, out_fpr_i, out_fnr_i = {}, {}, {}
    
    for i in sim_n_k:
        print(f'Simulating for n_k = {i} ...')
        out_sim_j, out_fpr_j, out_fnr_j = [], [], []
        
        for j in sim_runs:
            np.random.seed(j)
            B = A * np.random.uniform(0.1, 1, (q, q)) * np.random.choice([-1, 1], size=(q, q), replace=True)
            Sigma = np.identity(q)
            
            # Number of datasets (intervention contexts) and corresponding sample sizes
            I_cal_true = np.zeros((q, K), dtype=int)
            for k in range(1, K):
                np.random.seed(j)
                I_cal_true[np.random.choice(q, 4, replace=False), k] = 1
                
            sigma_int = np.repeat(0.1, K)
            
            n_all = np.repeat(i, K)
            
            out_data = mcmc.gen_dataset(
                A_true = A, B = B, Sigma = Sigma, I_cal = I_cal_true, sigma_int = sigma_int, n_all = n_all
                )
            out_mcmc = mcmc.mcmc_dag_targets(
                X = out_data['X'], S = S, burn = burn, w = w, a = None, a_k = a_k, b_k = b_k,\
                    n_all = n_all, seed = j, hide_pbar=True
                )
            out_sim_j.append(
                out_mcmc
            )
            out_fpr_j.append(
                fpr(I_cal_true, np.round(out_mcmc['Targets_estimate']))
            )
            out_fnr_j.append(
                fnr(I_cal_true, np.round(out_mcmc['Targets_estimate']))
            )
                
        out_sim_i[i], out_fpr_i[i], out_fnr_i[i] = out_sim_j, out_fpr_j, out_fnr_j
        
        # Clear the current cell's output
        clear_output(wait=True)

    sim_results = merge_sim_res(out_sim_i)
    fpr_results = merge_sim_res(out_fpr_i)
    fnr_results = merge_sim_res(out_fnr_i)

    shd_results = {}
    for i in sim_results:
        tmp_shd_list = []
        for j in sim_results[i]:
            tmp_shd = mcmc.shd(A, j['DAG_estimate'])
            tmp_shd_list.append(tmp_shd)
        shd_results[i] = tmp_shd_list
        

    return {
        'sim_results':sim_results,
        'shd':shd_results,
        'fpr': fpr_results,
        'fnr':fnr_results
            }
    
    
    
    