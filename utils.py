# -*- coding: utf-8 -*-
"""
Utility functions for Toeplitz Covariance Estimation
"""

import torch
import numpy as np
from scipy.linalg import sqrtm


def RSCM(x, M, mu=1, alpha=0.01):
    """Regularized Sample Covariance Matrix"""
    b, n, d = x.shape
    cov = x.swapaxes(2, 1) @ torch.conj(x) / M
    cov = mu * alpha * torch.eye(d)[None] + (1 - alpha) * cov
    return cov


def KL(ahat, a):
    """KL divergence between covariance matrices"""
    K = torch.linalg.pinv(ahat)
    v = .5 * (torch.real(torch.vmap(torch.trace)(K @ a)) - torch.linalg.slogdet(K @ a)[1] - a.shape[-1])
    return v.mean().item()


def RMSE(ahat, a):
    """Root Mean Squared Error for first row of Toeplitz matrices"""
    return torch.sum(torch.abs(ahat[:, 0, :] - a[:, 0, :]) ** 2, 1).mean()


def RMSE_VEC(ahat, a):
    """RMSE vector (per sample)"""
    return torch.sum(torch.abs(ahat[:, 0, :] - a[:, 0, :]) ** 2, 1)


def sample_from_spectrum(p, K):
    """Sample indices from spectrum"""
    p = p / p.sum()
    indices = torch.multinomial(p, K, replacement=True)
    return indices


def get_init_spect2(scm, K, N, L=1000, power=1.0):
    """Get initial spectrum with power weighting"""
    grid_o = torch.linspace(0, 2 * torch.pi, L)
    n = torch.arange(N).unsqueeze(1)  # (N, 1)
    A = torch.exp(1j * n * grid_o)  # (N, L)
    spectrum = torch.abs(torch.sum(A.conj() * (scm @ A), dim=0))
    spectrum = spectrum.pow(power)
    spectrum = spectrum / spectrum.sum()
    indices = sample_from_spectrum(spectrum, K)
    quantized_f = indices / L * 2 * torch.pi
    return quantized_f


def toeplitz_from_first_row(X):
    """Construct batch of Toeplitz matrices from first row"""
    B, D = X.shape
    i = torch.arange(D, device=X.device).view(-1, 1)  # shape (D, 1)
    j = torch.arange(D, device=X.device).view(1, -1)  # shape (1, D)
    idx = (j - i).abs()  # shape (D, D), values in [0, D-1]
    idx = idx.unsqueeze(0).expand(B, -1, -1)  # (B, D, D)
    b_idx = torch.arange(B, device=X.device).view(-1, 1, 1).expand(-1, D, D)  # (B, D, D)
    T = X[b_idx, idx]  # shape (B, D, D)
    return T


def quantize_indices(p, K):
    """Quantize probability distribution into K bins"""
    p = p / p.sum()
    F = torch.cumsum(p, dim=0)
    quantile_centers = (torch.arange(K, dtype=p.dtype) + 0.5) / K
    indices = torch.searchsorted(F, quantile_centers)
    return indices


def get_init_spect(scm, K, N, L=10000):
    """Get initial spectrum using quantization"""
    grid_o = torch.linspace(0, 2 * torch.pi, L)
    n = torch.arange(N).unsqueeze(1)  # (N, 1)
    A = torch.exp(1j * n * grid_o)  # (N, L)
    spectrum = torch.abs(torch.sum(A.conj() * (scm @ A), dim=0))
    spectrum = spectrum / spectrum.sum()
    quantized_f = quantize_indices(spectrum, K) / L * 2 * torch.pi
    return quantized_f


def generate_toeplitz_basis(m):
    """Generate Toeplitz basis matrices for CRB computation"""
    basis_matrices = []
    for g in range(1, m + 1):
        B = torch.zeros((m, m), dtype=torch.cfloat)
        for i in range(m):
            for k in range(m):
                if i - k == g - 1:
                    if g - 1 == 0:
                        B[i, k] = 1 + 1j
                    else:
                        B[i, k] = 1 - 1j
                elif k - i == g - 1:
                    B[i, k] = 1 + 1j
        basis_matrices.append(B)
    return basis_matrices


def compute_crb(R, n):
    """
    Compute the Cramér-Rao Bound for Toeplitz covariance matrices.

    Parameters:
        R (torch.Tensor): Toeplitz covariance matrix of shape (m, m).
        n (int): Number of independent samples.

    Returns:
        torch.Tensor: Sum of per-parameter CRBs (lower bound on MSE).
    """
    m = R.shape[0]
    P = m
    theta_dim = 2 * P - 1
    basis_matrices = generate_toeplitz_basis(P)
    R_inv = torch.linalg.inv(R)

    # Build derivatives dR/dθ
    dR_dtheta = []
    for i in range(P):
        dR_dtheta.append(basis_matrices[i].real)
    for i in range(1, P):
        dR_dtheta.append(1j * basis_matrices[i].imag)

    fim = torch.zeros((theta_dim, theta_dim), dtype=torch.complex64, device=R.device)

    # Slepian-Bangs formula
    for i in range(theta_dim):
        for j in range(theta_dim):
            dRi = dR_dtheta[i].to(torch.complex64)
            dRj = dR_dtheta[j].to(torch.complex64)
            term = R_inv @ dRi @ R_inv @ dRj
            fim[i, j] = torch.trace(term)

    fim_inv = torch.linalg.inv(fim)
    fim_inv = (1 / n) * fim_inv

    # Compute per-parameter CRB
    crb_params = torch.zeros(P, dtype=R.dtype, device=R.device)
    for i in range(P):
        if i == 0:
            crb_params[i] = fim_inv[i, i]
        else:
            crb_params[i] = fim_inv[i, i] + fim_inv[i + P - 1, i + P - 1]

    total_crb = torch.abs(crb_params).sum()
    return total_crb


def to_serializable(obj):
    """Convert numpy/torch objects to JSON-serializable types"""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    else:
        return obj


# Data generation functions

def atom_data(num_test, N, d=15, on_grid_freq=True):
    """Generate ATOM benchmark data"""
    torch.manual_seed(0)
    np.random.seed(0)

    omegas = torch.tensor(
        [0.2167, 0.6500, 1.0833, 1.3, 1.5166, 1.9500, 2.3833, 2.8166, 3.2499, 3.6832, 4.1166, 4.5499, 4.9832, 5.4165,
         5.8499][:d]).unsqueeze(0)
    if not on_grid_freq:
        omegas[:, 3] = 1.25
        omegas[:, 7] = 3.01
        omegas[:, 12] = 5.20
        omegas[:, 14] = 5.8

    amplitudes = torch.sqrt(torch.tensor(list(range(1, d + 1)), dtype=torch.cfloat).unsqueeze(0))
    grid_t = torch.arange(d)
    z = amplitudes[:, :, None] * torch.exp(1j * (grid_t[None, None, :] * omegas[:, :, None]))
    cn = np.sqrt(.5) * torch.randn(num_test, N, d) + 1j * np.sqrt(.5) * torch.randn(num_test, N, d)
    data = torch.einsum('esf,efd->esd', cn, z)
    cov = torch.einsum('efd,efr->edr', z, z.conj())
    return data, cov.repeat(num_test, 1, 1), omegas


def generate_data_toplitz_batch(num_examples=1, num_samples=20, dimension=6, num_freq=6, sigma2_noise=0.17,
                                ret_f=False, AA=2.5):
    """Generate random Toeplitz data"""
    torch.manual_seed(1)
    omegas = torch.sort(torch.rand(1, num_freq))[0] * 2 * np.pi
    amplitudes = torch.sort(torch.rand(1, num_freq) * AA)[0]
    print('omegas: ', omegas)
    print('true amp:', amplitudes)

    grid_t = torch.arange(dimension)
    z = amplitudes[:, :, None] * torch.exp(1j * (grid_t[None, None, :] * omegas[:, :, None]))
    cn = np.sqrt(.5) * torch.randn(num_examples, num_samples, num_freq) + 1j * np.sqrt(.5) * torch.randn(num_examples,
                                                                                                         num_samples,
                                                                                                         num_freq)
    noise = np.sqrt(.5) * torch.randn(num_examples, num_samples, dimension) + 1j * np.sqrt(.5) * torch.randn(
        num_examples, num_samples, dimension)
    data = torch.einsum('esf,efd->esd', cn, z) + sigma2_noise * noise
    cov = torch.einsum('efd,efr->edr', z, z.conj()) + sigma2_noise ** 2 * torch.eye(dimension)[None, :, :]

    if ret_f:
        return data, cov, omegas.squeeze(0)
    return data, cov


def gen_Gamma_varA(alpha, P):
    """Generate the precision matrix Gamma using Gohberg-Semencul"""
    from scipy.linalg import toeplitz
    B_m = TriaToepMulShort(alpha, P, False)
    C_m = TriaToepMulShort(alpha, P, True)
    return (1 / alpha[0]) * (B_m - C_m)


def TriaToepMulShort(c, P, rev):
    """Compute the matrix multiplication of a triangular Toeplitz matrix with its transpose"""
    if not rev:
        M = np.zeros((P, P), dtype=np.complex128)
        for i in range(len(c)):
            for j in range(len(c)):
                if i == 0:
                    M[i, i + j] = c[i] * c[i + j]
                    M[i + j, i] = M[i, i + j]
                else:
                    M[i, min(i + j, len(c) - 1)] = M[i - 1, min(i + j - 1, len(c) - 2)] + c[i] * c[
                        min(i + j, len(c) - 1)]
                    M[min(i + j, len(c) - 1), i] = M[i, min(i + j, len(c) - 1)]
        for i in range(len(c), P):
            for j in range(len(c)):
                M[min(i - j, P - 1), i] = M[min(i - j - 1, P - 2), i - 1]
                M[i, min(i - j, P - 1)] = M[min(i - j, P - 1), i]
        return M
    else:
        cflip = np.flip(c[1:])
        M = np.zeros((P, P), dtype=np.complex128)
        Mfull = np.zeros((len(c) - 1, len(c) - 1), dtype=np.complex128)
        for i in range(len(c) - 1):
            for j in range(i, len(c) - 1):
                if i == 0:
                    Mfull[i, j] = cflip[i] * cflip[j]
                    Mfull[j, i] = Mfull[i, j]
                else:
                    Mfull[i, j] = Mfull[i - 1, j - 1] + cflip[i] * cflip[j]
                    Mfull[j, i] = Mfull[i, j]
        M[-len(c) + 1:, -len(c) + 1:] = Mfull
        return M


def ar_roots_to_freqs(ar_coeffs, fs=1.0):
    """Given AR coefficients, return the angular frequencies of the complex poles"""
    a = np.asarray(ar_coeffs)
    poly = np.concatenate([[1], -a])
    roots = np.roots(poly)
    roots = roots[np.abs(roots) < 1 + 1e-6]
    angles = np.angle(roots)
    freqs = np.unique(np.abs(angles))
    return torch.from_numpy(freqs).to(torch.complex64)


def generate_AR_cov(N, sigma, a):
    """Generate AR process covariance"""
    a = np.asarray(a)
    p = len(a)
    B = np.eye(p)
    B = B[:-1, :]
    B = np.vstack([a, B])

    eigvals = np.linalg.eigvals(B)
    if np.all(np.abs(eigvals) < 1):
        print("stable AR process")

    a0 = 1 / (sigma ** 2)
    ar = -a * a0
    alpha = np.concatenate([[a0], ar])
    G = gen_Gamma_varA(alpha, N)
    C = np.linalg.inv(G)
    return C


def rlevinson(a, efinal):
    """Reverse Levinson recursion"""
    a = np.array(a)
    realdata = np.isrealobj(a)

    p = len(a)
    if p < 2:
        raise ValueError('Polynomial should have at least two coefficients')

    if realdata:
        U = np.zeros((p, p))
    else:
        U = np.zeros((p, p), dtype=np.complex128)
    U[:, p - 1] = np.conj(a[-1::-1])

    p = p - 1
    e = np.zeros(p)
    e[-1] = efinal

    # Step down
    for k in range(p - 1, 0, -1):
        [a, e[k - 1]] = levdown(a, e[k])
        U[:, k] = np.concatenate((np.conj(a[-1::-1].transpose()),
                                  [0] * (p - k)))

    e0 = e[0] / (1. - abs(a[1] ** 2))
    U[0, 0] = 1
    kr = np.conj(U[0, 1:])
    kr = kr.transpose()

    R = np.zeros(1, dtype=complex)
    k = 1
    R0 = e0
    R[0] = -np.conj(U[0, 1]) * R0

    # Actual recursion
    for k in range(1, p):
        r = -sum(np.conj(U[k - 1::-1, k]) * R[-1::-1]) - kr[k] * e[k - 1]
        R = np.insert(R, len(R), r)

    R = np.insert(R, 0, e0)
    return R


def levdown(anxt, enxt=None):
    """One step backward Levinson recursion"""
    anxt = anxt[1:]
    knxt = anxt[-1]
    if knxt == 1.0:
        raise ValueError('At least one of the reflection coefficients is equal to one.')

    acur = (anxt[0:-1] - knxt * np.conj(anxt[-2::-1])) / (1. - abs(knxt) ** 2)
    ecur = None
    if enxt is not None:
        ecur = enxt / (1. - np.dot(knxt.conj().transpose(), knxt))

    acur = np.insert(acur, 0, 1)
    return acur, ecur


def generate_ar1(num_examples=1, M=None, dimension=10):
    """Generate AR(3) process data"""
    torch.manual_seed(1)
    np.random.seed(1)

    a = [0.5, 0.2, 0.05]
    MM = generate_AR_cov(dimension, 0.8, a)

    true_freq = ar_roots_to_freqs(a)
    true_freq = torch.cat([true_freq, torch.rand(dimension - true_freq.shape[0]) * 2 * np.pi], 0)
    Msqrt = sqrtm(MM)

    real = np.sqrt(0.5) * np.random.randn(num_examples, M, dimension)
    imag = np.sqrt(0.5) * np.random.randn(num_examples, M, dimension)
    X0 = real + 1j * imag

    data = X0 @ Msqrt.T
    data = torch.tensor(data.astype(np.complex128), dtype=torch.complex64)
    true_covariances = torch.tensor(MM, dtype=torch.complex64).repeat(num_examples, 1, 1)
    return data, true_covariances, true_freq
