import os
import numpy as np
import matplotlib.pyplot as plt
from common import load_dataset, plot_true_sine, calculate_rmse

output_dir = "output part 3"
os.makedirs(output_dir, exist_ok=True)

def legendre_features(x: np.ndarray, degree: int = 9) -> np.ndarray:
    """Generates shifted Legendre polynomial basis P_j(2x-1)[cite: 4, 5]."""
    return np.polynomial.legendre.legvander(2 * x - 1, degree)

def run_part3():
    x_train, y_train = load_dataset('A1 - Data Sets/poly_train.csv')
    x_test, y_test = load_dataset('A1 - Data Sets/poly_test.csv')
    
    Phi_train = legendre_features(x_train)
    Phi_test = legendre_features(x_test)
    
    d = 9
    eta = 0.05
    iters = 20000
    lams = np.logspace(-4, 0, 21)
    
    # ---------------------------------------------------------
    # Subgradient Descent Function
    # ---------------------------------------------------------
    def run_subgradient_descent(lam):
        w = np.zeros(d + 1)
        best_E = float('inf')
        E_history = []
        best_E_history = []
        
        for _ in range(iters):
            E = 0.5 * np.sum((Phi_train @ w - y_train)**2) + lam * np.sum(np.abs(w))
            if E < best_E:
                best_E = E
            E_history.append(E)
            best_E_history.append(best_E)
            
            grad = Phi_train.T @ (Phi_train @ w - y_train) + lam * np.sign(w)
            w = w - eta * grad
            
        return w, E_history, best_E_history

    # ---------------------------------------------------------
    # Deliverable 1: Convergence and verification at lambda=0.1
    # ---------------------------------------------------------
    lam_01 = 0.1
    w_final_01, E_hist, best_E_hist = run_subgradient_descent(lam_01)
    
    plt.figure(figsize=(8, 6))
    plt.plot(E_hist, label='E(w)', alpha=0.5, color='orange')
    plt.plot(best_E_hist, label='Best-so-far E', color='red', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Objective E')
    plt.title('Deliverable 1: Lasso Convergence at $\lambda=0.1$')
    plt.legend()
    plt.grid(True, ls="--")
    plt.savefig(os.path.join(output_dir, 'deliverable_1_convergence.png'))
    plt.close()

    # Verification Math
    threshold = 2 * eta * lam_01
    zero_mask = np.abs(w_final_01) < threshold
    r = y_train - Phi_train @ w_final_01
    
    non_zero_cond = np.max(np.abs(Phi_train[:, ~zero_mask].T @ r - lam_01 * np.sign(w_final_01[~zero_mask]))) if np.sum(~zero_mask) > 0 else 0
    zero_cond = np.max(np.abs(Phi_train[:, zero_mask].T @ r) / lam_01) if np.sum(zero_mask) > 0 else 0
    
    w_star_ref = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])
    w_diff = np.linalg.norm(w_final_01 - w_star_ref)

    # ---------------------------------------------------------
    # Deliverables 2 & 3: RMSE Sweep and Coefficient Path
    # ---------------------------------------------------------
    w_all = []
    train_rmses, test_rmses, zero_counts = [], [], []
    
    for lam in lams:
        w_lam, _, _ = run_subgradient_descent(lam)
        w_all.append(w_lam)
        train_rmses.append(calculate_rmse(y_train, Phi_train @ w_lam))
        test_rmses.append(calculate_rmse(y_test, Phi_test @ w_lam))
        zero_counts.append(np.sum(np.abs(w_lam) < 2 * eta * lam))
        
    w_all = np.array(w_all)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lams, train_rmses, label='Train RMSE', marker='o')
    plt.plot(lams, test_rmses, label='Test RMSE', marker='s')
    plt.axhline(0.1, color='k', linestyle='--', label='Noise (0.1)')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('RMSE')
    plt.title('Deliverable 2: Lasso RMSE Sweep')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'deliverable_2_rmse_sweep.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    for j in range(d + 1):
        plt.plot(lams, w_all[:, j], label=f'w_{j}')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Coefficient Value')
    plt.title('Deliverable 3: Lasso Coefficient Path')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_3_coeff_path.png'))
    plt.close()

    # ---------------------------------------------------------
    # Deliverable 4 & 5: Selecting lambda* and Lasso vs Ridge
    # ---------------------------------------------------------
    opt_idx_lasso = np.argmin(test_rmses)
    lam_star_lasso = lams[opt_idx_lasso]
    w_star_lasso = w_all[opt_idx_lasso]
    zero_idx = np.where(np.abs(w_star_lasso) < 2 * eta * lam_star_lasso)[0]

    # Ridge in Legendre Basis
    test_rmses_ridge = []
    w_ridge_all = []
    I = np.eye(d + 1)
    for lam in lams:
        w_r = np.linalg.solve(Phi_train.T @ Phi_train + lam * I, Phi_train.T @ y_train)
        w_ridge_all.append(w_r)
        test_rmses_ridge.append(calculate_rmse(y_test, Phi_test @ w_r))
        
    opt_idx_ridge = np.argmin(test_rmses_ridge)
    lam_star_ridge = lams[opt_idx_ridge]
    w_star_ridge = w_ridge_all[opt_idx_ridge]

    # ---------------------------------------------------------
    # Deliverable 6: Fitted Curves
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    x_plot = np.linspace(0, 1, 200)
    Phi_plot = legendre_features(x_plot)
    plt.scatter(x_train, y_train, color='red', label='Train Data')
    plot_true_sine(plt.gca())
    plt.plot(x_plot, Phi_plot @ w_star_lasso, label=f'Lasso ($\lambda={lam_star_lasso:.4f}$)', color='blue')
    plt.plot(x_plot, Phi_plot @ w_star_ridge, label=f'Ridge ($\lambda={lam_star_ridge:.4f}$)', color='green', linestyle='-.')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.title('Deliverable 6: Lasso vs Ridge Fit at optimal $\lambda$')
    plt.savefig(os.path.join(output_dir, 'deliverable_6_fitted_curves.png'))
    plt.close()

    # Write Report
    with open(os.path.join(output_dir, 'part3_report.txt'), 'w') as f:
        f.write("Deliverable 1 Verification:\n")
        f.write(f"  max |Phi_j^T r - lam*sign(w_j)| for w!=0: {non_zero_cond:.6f}\n")
        f.write(f"  max |Phi_j^T r| / lam for w==0 (should be <=1): {zero_cond:.6f}\n")
        f.write(f"  ||w - w_star_ref||_2: {w_diff:.4f}\n\n")
        
        f.write("Deliverable 4: Selecting lambda*\n")
        f.write(f"  lambda* = {lam_star_lasso:.4f}\n")
        f.write(f"  Test RMSE = {test_rmses[opt_idx_lasso]:.6f}\n")
        f.write(f"  Indices of approx 0 coeffs: {zero_idx.tolist()}\n")
        f.write(f"  Total 0 coeffs: {len(zero_idx)}\n\n")
        
        f.write("Deliverable 5 & 7 Table (Lasso vs Ridge):\n")
        f.write(f"  Lasso (lam={lam_star_lasso:.4f}): RMSE={test_rmses[opt_idx_lasso]:.4f}, ||w||_2={np.linalg.norm(w_star_lasso):.4f}, |w_j|<0.01 count: {np.sum(np.abs(w_star_lasso)<0.01)}\n")
        f.write(f"  Ridge (lam={lam_star_ridge:.4f}): RMSE={test_rmses_ridge[opt_idx_ridge]:.4f}, ||w||_2={np.linalg.norm(w_star_ridge):.4f}, |w_j|<0.01 count: {np.sum(np.abs(w_star_ridge)<0.01)}\n")
        f.write("\nDiscussion (Deliverable 7):\n")
        f.write("Because the L1 norm creates a diamond-shaped constraint geometry, the objective contours hit the corners exactly on the axes, driving coefficients exactly to zero. In contrast, the L2 ball is circular, so contours meet it tangentially away from the axes, only shrinking coefficients without zeroing them. As lambda grows past optimal, both models over-penalize: test error rises as they underfit, and train error steadily climbs towards the variance of the targets.\n")

if __name__ == "__main__":
    run_part3()