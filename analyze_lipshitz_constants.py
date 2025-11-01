# -*- coding: utf-8 -*-
"""
Validate theoretical Hessian derivations against direct computation.
This checks if our formulas for H_aa and H_ww are correct.
"""

import numpy as np
import torch
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

torch.set_default_dtype(torch.float64)
DEVICE = "cpu"

# ============================================================
# CORE MODEL
# ============================================================

def softplus(a):
    return torch.nn.functional.softplus(a)

def softplus_grad(a):
    """s'(a) = sigmoid(a)"""
    return torch.sigmoid(a)

def softplus_hess(a):
    """s''(a) = sigmoid(a)(1-sigmoid(a))"""
    sig = torch.sigmoid(a)
    return sig * (1 - sig)

def build_V(P, omega):
    n = torch.arange(P, dtype=torch.float64, device=omega.device).reshape(-1,1)
    return torch.exp(1j * (n @ omega.reshape(1,-1)))

def C_from_aw(P, a, omega, eps=1e-3):
    V = build_V(P, omega)
    s = softplus(a)
    C = V @ (s.reshape(-1,1) * V.conj().T)
    C = (C + C.conj().T)/2 + eps * torch.eye(P, dtype=torch.complex128, device=a.device)
    return C

def complex_normal(P, N, device=DEVICE):
    z = torch.randn(P, N, dtype=torch.float64, device=device)
    w = torch.randn(P, N, dtype=torch.float64, device=device)
    return (z + 1j*w)/np.sqrt(2.0)

def sample_covariance(C_true, M):
    P = C_true.shape[0]
    chol = torch.linalg.cholesky(C_true)
    X = chol @ complex_normal(P, M, device=C_true.device)
    S = (X @ X.conj().T)/M
    return (S + S.conj().T)/2

def loss_L(S, a, omega, eps=1e-3):
    P = S.shape[0]
    C = C_from_aw(P, a, omega, eps)
    chol = torch.linalg.cholesky(C)
    X = torch.cholesky_solve(S, chol)
    term1 = torch.real(torch.trace(X))
    logdetC = 2.0 * torch.sum(torch.log(torch.real(torch.diagonal(chol))))
    return term1 + logdetC

# ============================================================
# THEORETICAL HESSIAN COMPUTATION
# ============================================================

def compute_theoretical_hessian_blocks(P, K, a, omega, S, eps=1e-3):
    """
    Compute H_aa and H_ww using derivatives - EXACT copy from simple_one_trial.py
    """
    V = build_V(P, omega)
    s = softplus(a)
    s_prime = softplus_grad(a)
    s_double_prime = softplus_hess(a)

    C = C_from_aw(P, a, omega, eps)
    C_inv = torch.linalg.inv(C)
    E = C_inv - C_inv @ S @ C_inv
    D = torch.diag(torch.arange(P, dtype=torch.float64, device=omega.device)).to(torch.complex128)

    # H_aa
    H_aa = torch.zeros(K, K, dtype=torch.float64)
    for i in range(K):
        for j in range(K):
            v_i = V[:, i:i+1]
            v_j = V[:, j:j+1]

            if i == j:
                term1 = s_double_prime[i] * torch.real(v_i.conj().T @ E @ v_i)
                H_aa[i, j] += float(term1.item())

            dC_daj = s_prime[j] * v_j @ v_j.conj().T
            term_a = C_inv @ dC_daj @ C_inv
            term_b = C_inv @ S @ C_inv
            dE_daj = -term_a + term_a @ S @ C_inv + term_b @ dC_daj @ C_inv
            quad_form = v_i.conj().T @ dE_daj @ v_i
            term2 = s_prime[i] * torch.real(quad_form)
            H_aa[i, j] += float(term2.item())

    # H_ww
    H_ww = torch.zeros(K, K, dtype=torch.float64)
    for i in range(K):
        v_i = V[:, i:i+1]
        Dv_i = D @ v_i

        for j in range(K):
            v_j = V[:, j:j+1]
            Dv_j = D @ v_j

            dC_dwj = s[j] * 1j * (Dv_j @ v_j.conj().T - v_j @ Dv_j.conj().T)
            Cinv_dC_Cinv = C_inv @ dC_dwj @ C_inv
            CSC = C_inv @ S @ C_inv
            dE_dwj = -Cinv_dC_Cinv + Cinv_dC_Cinv @ S @ C_inv + CSC @ dC_dwj @ C_inv

            if i == j:
                term1 = (1j * Dv_i).conj().T @ D @ E @ v_i
                term2 = v_i.conj().T @ D @ dE_dwj @ v_i
                term3 = v_i.conj().T @ D @ E @ (1j * Dv_i)
                H_ww[i, j] = 2 * s[i] * torch.imag(term1 + term2 + term3).item()
            else:
                quad_form = v_i.conj().T @ D @ dE_dwj @ v_i
                H_ww[i, j] = 2 * s[i] * torch.imag(quad_form).item()

    return H_aa, H_ww

# ============================================================
# EMPIRICAL HESSIAN VIA AUTOGRAD
# ============================================================

def compute_empirical_hessian_blocks(P, K, a, omega, S, eps=1e-3):
    """
    Compute H_aa and H_ww using finite differences on gradients.
    This matches the method used in verify_derivatives.py which gives R²=1.0
    """

    # Helper function to compute gradient
    def gradient_a(a_val, omega_val, S, eps):
        """Compute ∂L/∂a"""
        a_var = a_val.clone().requires_grad_(True)
        loss = loss_L(S, a_var, omega_val, eps)
        grad = torch.autograd.grad(loss, a_var)[0]
        return grad

    def gradient_omega(a_val, omega_val, S, eps):
        """Compute ∂L/∂ω"""
        omega_var = omega_val.clone().requires_grad_(True)
        loss = loss_L(S, a_val, omega_var, eps)
        grad = torch.autograd.grad(loss, omega_var)[0]
        return grad

    # Compute H_aa using finite differences
    h = 1e-7
    H_aa_empirical = torch.zeros(K, K, dtype=torch.float64)

    grad_a_base = gradient_a(a, omega, S, eps)
    for j in range(K):
        a_plus = a.clone()
        a_plus[j] += h
        grad_a_plus = gradient_a(a_plus, omega, S, eps)
        H_aa_empirical[:, j] = (grad_a_plus - grad_a_base) / h

    # Compute H_ww using finite differences
    H_ww_empirical = torch.zeros(K, K, dtype=torch.float64)

    grad_omega_base = gradient_omega(a, omega, S, eps)
    for j in range(K):
        omega_plus = omega.clone()
        omega_plus[j] += h
        grad_omega_plus = gradient_omega(a, omega_plus, S, eps)
        H_ww_empirical[:, j] = (grad_omega_plus - grad_omega_base) / h

    return H_aa_empirical, H_ww_empirical

# ============================================================
# MAIN VALIDATION
# ============================================================

def main_validation():
    """
    Validate theoretical formulas against autograd computation.
    """
    # Test configuration
    P = 8
    K = 16
    M = 300
    EPS = 1e-4
    N_TRIALS = 1000
    SEED = 1

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = []

    print("Validating Hessian formulas...")
    print("=" * 70)

    for trial in range(N_TRIALS):
        # Generate random problem
        a_true = torch.randn(K, device=DEVICE)
        w_true = 2 * pi * torch.rand(K, device=DEVICE)

        C_true = C_from_aw(P, a_true, w_true, EPS)
        S = sample_covariance(C_true, M)

        # Use estimated parameters
        a_est = torch.randn(K, device=DEVICE)
        w_est = 2 * pi * torch.rand(K, device=DEVICE)

        # Compute theoretical Hessian blocks
        H_aa_theory, H_ww_theory = compute_theoretical_hessian_blocks(P, K, a_est, w_est, S, EPS)

        # ------------------------------------------------------------
        # Proxy approximations for L_w
        # ------------------------------------------------------------
        C = C_from_aw(P, a_est, w_est, EPS)
        C_inv = torch.linalg.inv(C)
        E = C_inv - C_inv @ S @ C_inv

        Cinv_norm = torch.linalg.norm(C_inv, ord=2).item()
        E_norm = torch.linalg.norm(E, ord=2).item()
        s_norm = torch.linalg.norm(softplus(a_est)).item()
        s_max = softplus(a_est).max().item()

        proxy1  = 2 * (s_norm) * (P**2) * (Cinv_norm**2)
        proxy2 = 2 * s_max * P * (E_norm * Cinv_norm + Cinv_norm**3)
        proxy3 = 2 * s_max * (P**2) * (E_norm + s_norm * Cinv_norm**2)
        proxy4 = (P**1.5) *  (s_norm**2 * Cinv_norm**1.5) # 2 * s_max * (P**2) *  (E_norm +s_norm * Cinv_norm**1.5)


        # === AMPLITUDE PROXIES ===
        V = build_V(P, w_est)
        s = softplus(a_est)
        s_prime = softplus_grad(a_est)
        s_double_prime = softplus_hess(a_est)

        s_prime_norm = torch.norm(s_prime).item()
        s_prime_max = torch.max(s_prime).item()
        s_double_prime_max = torch.max(s_double_prime).item()

        proxyA1 = s_prime_max * P * (E_norm + Cinv_norm)
        proxyA2 = s_prime_max * P * (E_norm + Cinv_norm**2)
        proxyA3 = P * (Cinv_norm)
        proxyA4 = s_prime_norm * (P**2) * (Cinv_norm**2)



        # Compute empirical (ground truth) Hessian blocks using finite differences
        # This matches the method from simple_one_trial.py which gives perfect results
        H_aa_empirical, H_ww_empirical = compute_empirical_hessian_blocks(P, K, a_est, w_est, S, EPS)

        # Compute spectral norms (Lipschitz constants)
        L_a_theory = float(torch.linalg.norm(H_aa_theory, ord=2).item())
        L_w_theory = float(torch.linalg.norm(H_ww_theory, ord=2).item())

        L_a_empirical = float(torch.linalg.norm(H_aa_empirical, ord=2).item())
        L_w_empirical = float(torch.linalg.norm(H_ww_empirical, ord=2).item())

        # Sanity check: print first trial results
        if trial == 0:
            print(f"\nSanity check - Trial 0:")
            print(f"  L_w (theory):    {L_w_theory:.4e}")
            print(f"  L_w (empirical): {L_w_empirical:.4e}")
            print(f"  Ratio:           {L_w_theory/L_w_empirical:.6f}")
            print(f"  Should be ≈ 1.0 if formulas correct")
            print(f"\n  H_ww[0:2,0:2] theory:\n{H_ww_theory[0:2,0:2]}")
            print(f"\n  H_ww[0:2,0:2] empirical:\n{H_ww_empirical[0:2,0:2]}")

        # Compute Frobenius norm differences (measure of formula error)
        error_aa = float(torch.linalg.norm(H_aa_theory - H_aa_empirical, ord='fro').item())
        error_ww = float(torch.linalg.norm(H_ww_theory - H_ww_empirical, ord='fro').item())

        relative_error_aa = error_aa / (torch.linalg.norm(H_aa_empirical, ord='fro').item() + 1e-12)
        relative_error_ww = error_ww / (torch.linalg.norm(H_ww_empirical, ord='fro').item() + 1e-12)


        # Debug: check individual diagonal entries
        if trial == 0:
            print(f"\nDEBUG - Trial 0:")
            print(f"H_ww[0:3,0:3] (theory):")
            print(H_ww_theory[0:3, 0:3])
            print(f"\nH_ww[0:3,0:3] (empirical):")
            print(H_ww_empirical[0:3, 0:3])
            print(f"\nElement-wise ratio (theory/empirical):")
            for i in range(3):
                for j in range(3):
                    ratio = H_ww_theory[i,j] / (H_ww_empirical[i,j].item() + 1e-10)
                    print(f"  [{i},{j}]: {ratio:.3f}")


        results.append({
            'trial': trial,
            'L_a_theory': L_a_theory,
            'L_a_empirical': L_a_empirical,
            'L_w_theory': L_w_theory,
            'L_w_empirical': L_w_empirical,
            'error_aa_fro': error_aa,
            'error_ww_fro': error_ww,
            'rel_error_aa': relative_error_aa,
            'proxy1': proxy1,
            'proxy2': proxy2,
            'proxy3': proxy3,
            'proxy4': proxy4,
            'proxyA1': proxyA1,
            'proxyA2': proxyA2,
            'proxyA3': proxyA3,
            'proxyA4': proxyA4,
            'rel_error_ww': relative_error_ww,
        })

        if trial % 10 == 0:
            print(f"Trial {trial}/{N_TRIALS}...")

    df = pd.DataFrame(results)
    df.to_csv("hessian_validation.csv", index=False)
    print("\n✓ Saved: hessian_validation.csv")

    # ============================================================
    # ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n1. AMPLITUDE HESSIAN H_aa:")
    print(f"   Mean relative error: {df['rel_error_aa'].mean():.2%}")
    print(f"   Median relative error: {df['rel_error_aa'].median():.2%}")
    print(f"   Max relative error: {df['rel_error_aa'].max():.2%}")

    print(f"\n2. FREQUENCY HESSIAN H_ww:")
    print(f"   Mean relative error: {df['rel_error_ww'].mean():.2%}")
    print(f"   Median relative error: {df['rel_error_ww'].median():.2%}")
    print(f"   Max relative error: {df['rel_error_ww'].max():.2%}")

    print(f"\n3. LIPSCHITZ CONSTANTS COMPARISON:")
    print(f"   L_a (theory):    median={df['L_a_theory'].median():.2e}, range=[{df['L_a_theory'].min():.1e}, {df['L_a_theory'].max():.1e}]")
    print(f"   L_a (empirical): median={df['L_a_empirical'].median():.2e}, range=[{df['L_a_empirical'].min():.1e}, {df['L_a_empirical'].max():.1e}]")
    print(f"   L_w (theory):    median={df['L_w_theory'].median():.2e}, range=[{df['L_w_theory'].min():.1e}, {df['L_w_theory'].max():.1e}]")
    print(f"   L_w (empirical): median={df['L_w_empirical'].median():.2e}, range=[{df['L_w_empirical'].min():.1e}, {df['L_w_empirical'].max():.1e}]")

    # Check for systematic scaling
    ratio_Lw = df['L_w_theory'] / df['L_w_empirical']
    print(f"\n4. SYSTEMATIC SCALING CHECK:")
    print(f"   L_w ratio (theory/empirical): median={ratio_Lw.median():.3f}, mean={ratio_Lw.mean():.3f}")
    print(f"   → If median ≈ constant (e.g., 0.5 or 2.0), there's a missing factor")

    # Create validation plots
    create_validation_plots(df)


    # ============================================================
    # PROXY VALIDATION - 2 SUBPLOTS
    # ============================================================
    print("\n" + "="*70)
    print("PROXY VALIDATION - proxyA3 vs L_a and proxy4 vs L_w")
    print("="*70)

    def r2_score(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        return 1 - ss_res / ss_tot

    # Compute R² scores for the selected proxies
    r2_la_proxyA3 = r2_score(df['L_a_empirical'], df['proxyA3'])
    r2_lw_proxy4 = r2_score(df['L_w_empirical'], df['proxy4'])

    print(f"proxyA3 vs L_a (Empirical): R² = {r2_la_proxyA3:.4f}")
    print(f"proxy4 vs L_w (Empirical): R² = {r2_lw_proxy4:.4f}")

    # Create 2-subplot figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: La upper bound with proxyA3
    axs[0].scatter(df['L_a_empirical'], df['proxyA3'], s=30, alpha=0.6, c='blue')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    # Perfect agreement line
    min_val = min(df['L_a_empirical'].min(), df['proxyA3'].min())
    max_val = max(df['L_a_empirical'].max(), df['proxyA3'].max())
    axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')

    axs[0].set_xlabel("$L_a$ (Empirical)", fontsize=12, fontweight='bold')
    axs[0].set_ylabel(r"$P \cdot \|C^{-1}\|$", fontsize=12, fontweight='bold')
    axs[0].set_title("$L_a$ Approximation", fontsize=13, fontweight='bold')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3, which='both')

    # Subplot 2: Lw with proxy4
    axs[1].scatter(df['L_w_empirical'], df['proxy4'], s=30, alpha=0.6, c='green')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    # Perfect agreement line
    min_val = min(df['L_w_empirical'].min(), df['proxy4'].min())
    max_val = max(df['L_w_empirical'].max(), df['proxy4'].max())
    axs[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')

    axs[1].set_xlabel("$L_ω$ (Empirical)", fontsize=12, fontweight='bold')
    axs[1].set_ylabel(r"$P^{1.5} \cdot \|s\|^2 \cdot \|C^{-1}\|^{1.5}$", fontsize=12, fontweight='bold')
    axs[1].set_title("$L_ω$ Approximation", fontsize=13, fontweight='bold')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("proxy_validation_selected.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: proxy_validation_selected.png")




    return df

# ============================================================
# VISUALIZATION
# ============================================================

def create_validation_plots(df):
    """Create scatter plots comparing theoretical vs empirical."""

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # Plot 1: L_a comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(df['L_a_empirical'], df['L_a_theory'], s=30, alpha=0.6, c='blue')

    # Perfect agreement line
    min_val = min(df['L_a_empirical'].min(), df['L_a_theory'].min())
    max_val = max(df['L_a_empirical'].max(), df['L_a_theory'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect agreement')

    ax1.set_xlabel("$L_a$ (Empirical - Autograd)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("$L_a$ (Theoretical Formula)", fontsize=12, fontweight='bold')
    ax1.set_title("Amplitude Lipschitz Constant\nValidation", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Compute R²
    ss_res = np.sum((df['L_a_empirical'] - df['L_a_theory'])**2)
    ss_tot = np.sum((df['L_a_empirical'] - df['L_a_empirical'].mean())**2)
    r2_a = 1 - ss_res / ss_tot
    ax1.text(0.05, 0.95, f"$R^2 = {r2_a:.4f}$", transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: L_w comparison
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(df['L_w_empirical'], df['L_w_theory'], s=30, alpha=0.6, c='green')

    min_val = min(df['L_w_empirical'].min(), df['L_w_theory'].min())
    max_val = max(df['L_w_empirical'].max(), df['L_w_theory'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect agreement')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("$L_ω$ (Empirical - Autograd)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("$L_ω$ (Theoretical Formula)", fontsize=12, fontweight='bold')
    ax2.set_title("Frequency Lipschitz Constant\nValidation", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    # Compute R² in log space
    log_emp = np.log10(df['L_w_empirical'])
    log_theory = np.log10(df['L_w_theory'])
    ss_res = np.sum((log_emp - log_theory)**2)
    ss_tot = np.sum((log_emp - log_emp.mean())**2)
    r2_w = 1 - ss_res / ss_tot
    ax2.text(0.05, 0.95, f"$R^2 = {r2_w:.4f}$ (log)", transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig("hessian_validation.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved: hessian_validation.png")
    plt.show()

    # Print conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if r2_a > 0.99 and r2_w > 0.99:
        print("✓ Theoretical formulas are CORRECT (R² > 0.99 for both)")
        print("  → Can proceed with using these formulas for bounds")
    elif r2_a > 0.95 and r2_w > 0.95:
        print("⚠ Theoretical formulas have small errors (R² > 0.95)")
        print("  → Formulas are approximately correct but may need refinement")
    else:
        print("✗ Theoretical formulas have SIGNIFICANT errors (R² < 0.95)")
        print("  → Need to check derivations carefully")

if __name__ == "__main__":
    df = main_validation()