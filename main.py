# -*- coding: utf-8 -*-
"""
Main experiment runner for Toeplitz Covariance Estimation
"""

import torch
import numpy as np
import os
import time
import json
from scipy.io import savemat

from utils import (
    KL, RMSE, compute_crb, to_serializable,
    atom_data, generate_data_toplitz_batch, generate_ar1
)
from GD import nnls_gd_backtracking_batch


def run_single_experiment(experiment_name, data_fn, save_suffix, num_test, Ms, Ks, KsA, dim, init,
                          device='cpu', save_data=False, results_dir='results_logs'):
    """
    Run a single experiment with various configurations.
    
    Parameters:
    - experiment_name: Name of the experiment
    - data_fn: Function to generate data
    - save_suffix: Suffix for saving results
    - num_test: Number of test samples
    - Ms: List of sample sizes to test
    - Ks: List of K values (multipliers for dimension) for GD with freq learning
    - KsA: List of K values for GD with amplitude only
    - dim: Dimension of the covariance matrix
    - init: Initialization method ('nnls', 'random', 'zeros')
    - device: torch device
    - save_data: Whether to save the data to .mat files
    - results_dir: Directory to save results
    
    Returns:
    - None (saves results to file)
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize result dictionaries
    KLs_gd, MSEs_gd, time_gd = {k: [] for k in Ks}, {k: [] for k in Ks}, {k: [] for k in Ks}
    KLs_gda, MSEs_gda, time_gda = {k: [] for k in KsA}, {k: [] for k in KsA}, {k: [] for k in KsA}
    crb_l = []

    for M in Ms:
        print(f'[{experiment_name}] dim={dim} | M={M}')
        data = data_fn(num_test, M, dim)
        x, cov = data[:2]
        true_freq = data[2] if len(data) > 2 else None

        if save_data:
            os.makedirs('test_mat', exist_ok=True)
            savemat(f'test_mat/{experiment_name}_M_{M}.mat', {
                'x_test': x.detach().numpy(),
                'cov_test': cov[0].detach().numpy()
            })

        batch_size = x.shape[0]
        crb_value = compute_crb(cov[0], M)
        print(f'CRB: {crb_value}')

        # Run GD with frequency + amplitude learning
        for k in Ks:
            K_val = int(k * dim)
            print(f'Running GD (learn amp+freq) K={K_val}')
            start = time.time()
            c_gd = nnls_gd_backtracking_batch(x.to(device), init=init, K=K_val, device=device)
            elapsed = (time.time() - start) / batch_size
            KLs_gd[k].append(KL(c_gd, cov))
            MSEs_gd[k].append(RMSE(c_gd, cov))
            time_gd[k].append(elapsed)
            print(f'  MSE: {MSEs_gd[k][-1]:.6f}, KL: {KLs_gd[k][-1]:.6f}, Time: {elapsed:.4f}s')

        # Run GDA with amplitude only learning
        for k in KsA:
            K_val = int(k * dim)
            print(f'Running GDA (amp only) K={K_val}')
            start = time.time()
            c_gda = nnls_gd_backtracking_batch(
                x.to(device), init=init, K=K_val,
                learn_freq=False, device=device
            )
            elapsed = (time.time() - start) / batch_size
            KLs_gda[k].append(KL(c_gda, cov))
            MSEs_gda[k].append(RMSE(c_gda, cov))
            time_gda[k].append(elapsed)
            print(f'  MSE: {MSEs_gda[k][-1]:.6f}, KL: {KLs_gda[k][-1]:.6f}, Time: {elapsed:.4f}s')

        crb_l.append(crb_value)

    # Prepare results dictionary
    save_dict = {
        'Ms': Ms,
        'dim': dim,
        'num_test': num_test,
        'KLs_gd': KLs_gd,
        'MSEs_gd': MSEs_gd,
        'time_gd': time_gd,
        'KLs_gda': KLs_gda,
        'MSEs_gda': MSEs_gda,
        'time_gda': time_gda,
        'crb': crb_l,
    }

    print('\n=== Final Results ===')
    print(f'Experiment: {experiment_name}')
    print(f'Dimension: {dim}, Num samples: {num_test}')
    print(f'MSEs_gd: {MSEs_gd}')
    print(f'MSEs_gda: {MSEs_gda}')
    print(f'CRBs: {crb_l}')

    # Save results to file
    save_path = os.path.join(results_dir, f'results_{save_suffix}_dim{dim}_{init}_{num_test}.txt')
    with open(save_path, 'w') as f:
        json.dump(to_serializable(save_dict), f, indent=2)
    print(f'\nSaved results to {save_path}')


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f'Using device: {device}')

    # Create results directory
    results_dir = 'results_logs'
    os.makedirs(results_dir, exist_ok=True)

    # Experiment configuration
    exps = [1, 2, 3]  # Which experiments to run: 1=Random Toeplitz, 2=ATOM On-Grid, 3=AR(1)
    Ms1 = list(range(10, 100, 10))  # Sample sizes
    Ks = [2, 4]  # K values for GD (learns freq + amp)
    KsA = []  # K values for GDA (learns amp only)
    dim = 15  # Dimension
    num_test = 100  # Number of test samples
    for Ms in [Ms1]:
        for init in ['random']:  # Initialization method
            if 1 in exps:
                # Random Toeplitz experiment
                AA = 2.5  # Amplitude parameter
                run_single_experiment(
                    experiment_name='Random Toeplitz',
                    data_fn=lambda num_test, M, dim: generate_data_toplitz_batch(
                        num_examples=num_test, num_samples=M,
                        dimension=dim, num_freq=dim,
                        ret_f=True, AA=AA
                    ),
                    save_suffix=f'exp1_{AA}_differentParamsLr',
                    num_test=num_test, Ms=Ms, Ks=Ks, KsA=KsA, dim=dim, init=init,
                    device=device, results_dir=results_dir
                )

            if 2 in exps:
                # ATOM On-Grid experiment
                run_single_experiment(
                    experiment_name='ATOM On-Grid',
                    data_fn=lambda num_test, M, dim: atom_data(num_test=num_test, N=M, d=dim),
                    save_suffix=f'exp2_differentParamsLr_highM',
                    num_test=num_test, Ms=Ms, Ks=Ks, KsA=KsA, dim=dim, init=init,
                    device=device, results_dir=results_dir
                )

            if 3 in exps:
                # AR(1) process experiment
                run_single_experiment(
                    experiment_name='AR(1) Data',
                    data_fn=lambda num_test, M, dim: generate_ar1(num_examples=num_test, M=M, dimension=dim),
                    save_suffix=f'exp3_differentParamsLr',
                    num_test=num_test, Ms=Ms, Ks=Ks, KsA=KsA, dim=dim, init=init,
                    device=device, results_dir=results_dir
                )

    print('\nAll experiments completed!')
