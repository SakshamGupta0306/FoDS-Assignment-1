import os
import numpy as np
import matplotlib.pyplot as plt
from common import load_dataset, plot_true_sine, generate_polynomial_features

output_dir = "output part 4"
os.makedirs(output_dir, exist_ok=True)

def run_part4():
    d = 3
    alpha = 0.05
    beta = 100.0  # 1 / 0.01
    lam_map = alpha / beta
    
    # ---------------------------------------------------------
    # Deliverables 1 & 2: Predictive Plot and Equivalence Check
    # ---------------------------------------------------------
    x_train_10, y_train_10 = load_dataset('A1 - Data Sets/p4_train_N10.csv')
    Phi_10 = generate_polynomial_features(x_train_10, d)
    
    # MAP / Ridge Solution
    w_map = np.linalg.solve(Phi_10.T @ Phi_10 + lam_map * np.eye(d + 1), Phi_10.T @ y_train_10)
    
    # Bayesian Posterior Mean and Covariance
    S_N = np.linalg.inv(alpha * np.eye(d + 1) + beta * Phi_10.T @ Phi_10)
    m_N = beta * S_N @ Phi_10.T @ y_train_10
    
    x_plot = np.linspace(0, 1, 200)
    Phi_plot = generate_polynomial_features(x_plot, d)
    
    # Vectorized Predictive distribution
    m_x = Phi_plot @ m_N
    # s^2(x) = 1/beta + phi(x)^T S_N phi(x). Diag computes this for all rows efficiently.
    s2_x = 1.0 / beta + np.sum((Phi_plot @ S_N) * Phi_plot, axis=1) 
    s_x = np.sqrt(s2_x)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train_10, y_train_10, color='red', label='Train Data (N=10)')
    plot_true_sine(plt.gca())
    plt.plot(x_plot, m_x, color='blue', label='Predictive Mean m(x)')
    plt.fill_between(x_plot, m_x - 1.96 * s_x, m_x + 1.96 * s_x, color='blue', alpha=0.2, label='95% Predictive Interval')
    
    # Overlay MAP
    plt.plot(x_plot, Phi_plot @ w_map, color='orange', linestyle=':', linewidth=3, label='MAP Estimate (Ridge)')
    plt.title('Deliverable 1 & 2: Bayesian Predictive Distribution (N=10)')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.savefig(os.path.join(output_dir, 'deliverable_1_2_predictive.png'))
    plt.close()

    # ---------------------------------------------------------
    # Deliverable 3: Effect of N
    # ---------------------------------------------------------
    N_list = [5, 10, 20, 50]
    
    plt.figure(figsize=(10, 6))
    
    for N in N_list:
        x_train, _ = load_dataset(f'A1 - Data Sets/p4_train_N{N}.csv')
        Phi_train = generate_polynomial_features(x_train, d)
        
        S_N = np.linalg.inv(alpha * np.eye(d + 1) + beta * Phi_train.T @ Phi_train)
        
        s2_x = 1.0 / beta + np.sum((Phi_plot @ S_N) * Phi_plot, axis=1)
        interval_width = 2 * 1.96 * np.sqrt(s2_x)
        
        plt.plot(x_plot, interval_width, label=f'N = {N}')

    irreducible_floor = 2 * 1.96 * np.sqrt(1.0 / beta)
    plt.axhline(irreducible_floor, color='black', linestyle='--', label='Irreducible Noise Floor')
    
    plt.xlabel('x')
    plt.ylabel('Predictive Interval Width (95%)')
    plt.title('Deliverable 3: Shrinking Uncertainty as N Grows')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'deliverable_3_N_sweep.png'))
    plt.close()
    
    # Write Report
    with open(os.path.join(output_dir, 'part4_report.txt'), 'w') as f:
        f.write("Deliverable 2: Equivalence Check\n")
        f.write(f"  ||w_MAP - m_N||_2 = {np.linalg.norm(w_map - m_N):.6e}\n")
        f.write("  This confirms the MAP point estimate perfectly matches the mean of the Bayesian predictive posterior.\n\n")
        
        f.write("Deliverable 4: Discussion\n")
        f.write("As N increases, the posterior covariance matrix S_N physically shrinks because the data contributes more information (beta * Phi^T Phi dominates), reducing the parameter uncertainty term phi(x)^T S_N phi(x). However, the irreducible observation-noise term (beta^-1) represents the inherent variance in the physical process generating the data, meaning predictions will always have a baseline uncertainty floor regardless of how accurately we estimate the polynomial weights.\n")

if __name__ == "__main__":
    run_part4()