# -*- coding: utf-8 -*-
"""
Gradient Descent Algorithms for Toeplitz Covariance Estimation
"""

import torch
import numpy as np

# Algorithm 1 in the paper
def gradient_descent_backtracking(
        objective_fn, params, alpha=0.5, beta=0.5, initial_lr=8e-2,
        num_iterations=2000, tol_grad=1e-6, tol_loss=1e-6, patience=12, verbose=True
):
    """
    Gradient descent with backtracking and stagnation-based early stopping.
    This is the standard GD with a single learning rate for all parameters.

    Parameters:
    - objective_fn: Callable returning scalar loss.
    - params: List of torch.Tensors to optimize.
    - alpha, beta: Backtracking line search parameters.
    - initial_lr: Initial learning rate.
    - num_iterations: Max number of iterations.
    - tol_grad, tol_loss: Tolerances for convergence.
    - patience: How many iterations with small change before stopping.
    - verbose: Print progress or not.

    Returns:
    - params: Optimized parameters.
    """

    prev_grad_norm = None
    prev_loss = None
    stable_steps = 0

    for i in range(num_iterations):
        for param in params:
            param.requires_grad_(True)
            if param.grad is not None:
                param.grad.zero_()

        loss = objective_fn(params)
        loss.backward()

        grad_norm = sum(param.grad.norm() ** 2 for param in params).sqrt()

        if prev_grad_norm is not None:
            grad_change = abs(grad_norm.item() - prev_grad_norm)
            loss_change = abs(loss.item() - prev_loss)

            if grad_change < tol_grad and loss_change < tol_loss:
                stable_steps += 1
                if stable_steps >= patience:
                    if verbose:
                        print(f"Stopping at iteration {i + 1}: small change in grad/loss for {patience} steps")
                    break
            else:
                stable_steps = 0

        prev_grad_norm = grad_norm.item()
        prev_loss = loss.item()

        # Backtracking line search
        lr = initial_lr
        with torch.no_grad():
            for param in params:
                param.data -= lr * param.grad
            new_loss = objective_fn(params)

            done = False
            beta_trials = 0
            while not done and new_loss > loss - alpha * lr * grad_norm ** 2:
                for param in params:
                    param.data += lr * param.grad  # Undo update
                lr *= beta
                beta_trials += 1
                if beta_trials > 20:
                    done = True
                for param in params:
                    param.data -= lr * param.grad
                new_loss = objective_fn(params)

    return params


# Algorithm 2 in the paper
def gradient_descent_backtracking_separate_lr(
        objective_fn, params, alpha=0.5, beta=0.5,
        lr_freq=9e-3, lr_amp=8e-2,  # separate lrs
        num_iterations=10000, tol_grad=1e-6, tol_loss=1e-6,
        patience=12, verbose=True
):
    """
    Gradient descent with backtracking, early stopping, and separate learning rates.
    This version uses different learning rates for frequencies and amplitudes.

    params[0]: frequencies (or detached if fixed)
    params[1]: amplitudes/weights
    """

    prev_grad_norm = None
    prev_loss = None
    stable_steps = 0

    for i in range(num_iterations):
        for param in params:
            param.requires_grad_(True)
            if param.grad is not None:
                param.grad.zero_()

        loss = objective_fn(params)
        loss.backward()

        grad_norm = sum(param.grad.norm() ** 2 for param in params).sqrt()

        if prev_grad_norm is not None:
            grad_change = abs(grad_norm.item() - prev_grad_norm)
            loss_change = abs(loss.item() - prev_loss)
            if grad_change < tol_grad and loss_change < tol_loss:
                stable_steps += 1
                if stable_steps >= patience:
                    if verbose:
                        print(f"Stopping at iteration {i + 1} due to small grad/loss change.")
                    break
            else:
                stable_steps = 0

        prev_grad_norm = grad_norm.item()
        prev_loss = loss.item()

        lr_f, lr_a = lr_freq, lr_amp
        f, a = params

        with torch.no_grad():
            # Initial step
            f.data -= lr_f * f.grad
            a.data -= lr_a * a.grad
            new_loss = objective_fn([f, a])

            # Backtracking loop
            beta_trials = 0
            while new_loss > loss - alpha * (lr_f * f.grad.norm() ** 2 + lr_a * a.grad.norm() ** 2):
                # Undo
                f.data += lr_f * f.grad
                a.data += lr_a * a.grad

                lr_f *= beta
                lr_a *= beta
                beta_trials += 1
                if beta_trials > 20:
                    break

                # Try new step
                f.data -= lr_f * f.grad
                a.data -= lr_a * a.grad
                new_loss = objective_fn([f, a])

    return params


def generate_toeplitz_covariance(frequencies, weights, D, device='cpu'):
    """
    Generate Toeplitz covariance matrix from frequencies and weights.
    
    Parameters:
    - frequencies: torch.Tensor of frequencies
    - weights: torch.Tensor of weights (amplitudes)
    - D: Dimension of the matrix
    - device: torch device
    
    Returns:
    - Toeplitz covariance matrix
    """
    indices = torch.arange(D, dtype=torch.float32, device=device)
    abs_diff = indices.view(-1, 1) - indices.view(1, -1)
    E = torch.exp(1j * frequencies[:, None, None] * abs_diff)
    toeplitz_matrix = 20 * torch.sum(torch.nn.functional.softplus(weights[:, None, None]) * E, dim=0)
    return toeplitz_matrix + 1e-3 * torch.eye(D, dtype=torch.complex64).to(device)


def log_likelihood(covariance_matrix, scm):
    """
    Compute log-likelihood for Gaussian distribution.
    
    Parameters:
    - covariance_matrix: Estimated covariance
    - scm: Sample covariance matrix
    
    Returns:
    - Log-likelihood value
    """
    inv_cov = torch.inverse(covariance_matrix)
    log_det_cov = torch.logdet(covariance_matrix)
    likelihood = torch.real(torch.trace(inv_cov @ scm) + log_det_cov)
    return likelihood


def approx_nnls(A, b):
    """
    Approximate Non-Negative Least Squares solution.
    
    Parameters:
    - A: Design matrix
    - b: Target vector
    
    Returns:
    - x: Approximate NNLS solution
    """
    x = torch.linalg.pinv(A) @ b
    ind = x > 0
    AA = A[:, ind]
    xx = torch.linalg.pinv(AA) @ b
    xx = xx * (xx >= 0) + 1e-3
    x = 0 * x
    x[ind] = xx
    return x


def nnls_gd_backtracking(scm, K, D=15, alpha=0.3, beta=0.5, init='nnls', num_iterations=45000, omegas=None,
                           learn_freq=True, return_params=False, device='cpu'):
    """
    GD with backtracking for Toeplitz covariance estimation.
    Version 2: Frequencies can be learned or fixed.
    
    Parameters:
    - scm: Sample covariance matrix
    - K: Number of frequencies
    - D: Dimension
    - alpha, beta: Backtracking parameters
    - init: Initialization method ('nnls', 'zeros', 'random')
    - num_iterations: Max iterations
    - omegas: Initial frequencies (optional)
    - learn_freq: Whether to learn frequencies
    - return_params: Whether to return parameters
    - device: torch device
    
    Returns:
    - Estimated covariance matrix (and optionally parameters)
    """
    frequencies = torch.linspace(0.00, 1, K, dtype=torch.float32, device=device) * 2 * np.pi
    if omegas is not None:
        frequencies = omegas
        frequencies = frequencies.detach()
    if not learn_freq:
        frequencies = frequencies.detach()

    indices = torch.arange(D, dtype=torch.float32).to(device)
    abs_diff = indices.view(-1, 1) - indices.view(1, -1)
    E = torch.exp(1j * frequencies[:, None, None] * abs_diff).reshape(K, D * D)
    E = torch.cat((E.real, E.imag), 1).T

    y = 1.0 * scm.reshape(D * D)
    y = torch.cat((y.real, y.imag))
    
    if init == 'nnls':
        weights = approx_nnls(E, y)
        weights = torch.log(torch.exp(weights / 20) - 1).to(torch.float32)  # inverse softplus
        beta = 0.6
    elif init == 'zeros':
        weights = torch.zeros(K, dtype=torch.float32, device=device)
    elif init == 'random':
        weights = (0.005 * torch.rand(K, device=device) * torch.trace(scm).real) / K
        weights = torch.log(torch.exp(weights / 20) - 1).to(torch.float32)

    def objective_fn(params):
        if learn_freq:
            covariance_matrix = generate_toeplitz_covariance(params[0], params[1], D, device)
        else:
            covariance_matrix = generate_toeplitz_covariance(frequencies, params[0], D, device)
        return log_likelihood(covariance_matrix, scm)

    if learn_freq:
        freqs_opt, weights_opt = gradient_descent_backtracking_separate_lr(
            objective_fn, [frequencies, weights],
            alpha=alpha, beta=beta, num_iterations=num_iterations
        )
    else:
        freqs_opt = frequencies
        weights_opt = gradient_descent_backtracking(
            objective_fn, [weights],
            alpha=alpha, beta=beta, num_iterations=num_iterations
        )[0]
    
    covariance_matrix = generate_toeplitz_covariance(freqs_opt, weights_opt, D, device)

    if return_params:
        weights_opt = 20 * torch.nn.functional.softplus(weights_opt)
        return covariance_matrix, freqs_opt.detach().cpu().numpy(), weights_opt.detach().cpu().numpy()
    return covariance_matrix


def nnls_gd_backtracking_batch(x, K, init='random', covv=None, omegas=None, learn_freq=True, device='cpu'):
    """
    Run GD optimization in batch mode.
    
    Parameters:
    - x: Input data (batch_size, num_samples, dimension)
    - K: Number of frequencies
    - init: Initialization method
    - covv: Optional pre-computed covariance
    - omegas: Optional initial frequencies
    - learn_freq: Whether to learn frequencies
    - device: torch device
    
    Returns:
    - Batch of estimated covariance matrices
    """
    from utils import RSCM
    
    covs = []
    M = x.shape[1]
    D = x.shape[2]
    for i in range(x.shape[0]):
        xi = x[i, :, :]
        scm = xi.T @ xi.conj() / M
        scm = scm.to(device)
        if covv is not None:
            scm = covv
        if omegas is not None:
            omegas = omegas.clone().detach()
        covs.append(
            nnls_gd_backtracking(scm, K, D, init=init, omegas=omegas, learn_freq=learn_freq, device=device).unsqueeze(0).detach()
        )
    cov = torch.cat(covs, 0)
    return cov

