# Estimation of Toeplitz Covariance Matrices using Overparameterized Gradient Descent

**Daniel Busbib and Ami Wiesel**  
School of Computer Science and Engineering  
The Hebrew University of Jerusalem

---

## Abstract

We consider covariance estimation under Toeplitz structure. Numerous sophisticated optimization methods have been developed to maximize the Gaussian log-likelihood under Toeplitz constraints. In contrast, recent advances in deep learning demonstrate the surprising power of simple gradient descent (GD) applied to overparameterized models. Motivated by this trend, we revisit Toeplitz covariance estimation through the lens of overparameterized GD. We model the $P\times P$ covariance as a sum of $K$ complex sinusoids with learnable parameters and optimize them via GD. We show that when $K = P$, GD may converge to suboptimal solutions. However, mild overparameterization ($K = 2P$ or $4P$) consistently enables global convergence from random initializations. We further propose an accelerated GD variant with separate learning rates for amplitudes and frequencies. When frequencies are fixed and only amplitudes are optimized, we prove that the optimization landscape is asymptotically benign and any stationary point recovers the true covariance. Finally, numerical experiments demonstrate that overparameterized GD can match or exceed the accuracy of state-of-the-art methods in challenging settings, while remaining simple and scalable.

---

## Algorithms

### Algorithm 1: Standard Gradient Descent with Backtracking

This algorithm optimizes both the frequencies and amplitudes of the complex sinusoids using standard gradient descent with a single learning rate. It employs backtracking line search to adaptively determine step sizes and includes early stopping based on gradient and loss stagnation. This is the baseline approach that works well with overparameterization.

**Key features:**
- Single learning rate for all parameters
- Armijo backtracking line search for adaptive step sizes
- Convergence detection via gradient/loss monitoring

### Algorithm 2: Accelerated Gradient Descent with Separate Learning Rates

An enhanced version that uses **separate learning rates** for frequencies and amplitudes. This recognizes that these two parameter types may require different step sizes for optimal convergence. The algorithm maintains the backtracking strategy but applies it independently to each parameter group, leading to faster and more stable convergence in practice.

**Key features:**
- Separate learning rates: smaller for frequencies, larger for amplitudes
- Independent backtracking for each parameter type
- Improved convergence speed and stability

---

## File Structure

- **`GD.py`** - Core gradient descent algorithms (Algorithm 1 and Algorithm 2)
- **`utils.py`** - Utility functions: metrics (KL, RMSE), Cramér-Rao Bound computation, data generation
- **`main.py`** - Experiment runner for reproducing paper results
- **`analyze_lipshitz_constants.py`** - Hessian validation and Lipschitz constant analysis (Figure 1)
- **`requirements.txt`** - Python dependencies

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```python
import torch
from utils import atom_data, KL, RMSE, compute_crb
from GD import nnls_gd_backtracking_batch

# Generate data
x, cov, omegas = atom_data(num_test=100, N=50, d=15)

# Run overparameterized GD (K = 2*dim or K = 4*dim)
K = 30  # Overparameterization: K = 2*d
cov_estimate = nnls_gd_backtracking_batch(
    x, K=K, init='random', learn_freq=True, device='cpu'
)

# Evaluate
print(f'KL divergence: {KL(cov_estimate, cov):.4f}')
print(f'RMSE: {RMSE(cov_estimate, cov):.4f}')
print(f'CRB: {compute_crb(cov[0], M=50):.4f}')
```

### Running Experiments

To reproduce the paper experiments:

```bash
python main.py
```

Results will be saved to `results_logs/` directory.

### Analyzing Lipschitz Constants (Figure 1)

To validate theoretical Hessian formulas and analyze Lipschitz constants:

```bash
python analyze_lipshitz_constants.py
```

This script:
- Validates theoretical Hessian formulas against numerical computation
- Computes Lipschitz constants for amplitude and frequency parameters
- Generates validation plots (`hessian_validation.png`, `proxy_validation_selected.png`)
- Saves results to `hessian_validation.csv`

---

## Citation

**Coming soon**

Note: A preliminary version of this work was presented at IEEE-CAMSAP 2025.

---


## Contact

- Daniel Busbib: daniel.busbib@mail.huji.ac.il
- Ami Wiesel: ami.wiesel@mail.huji.ac.il
