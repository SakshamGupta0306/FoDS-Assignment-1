import os
import numpy as np
import matplotlib.pyplot as plt
from common import load_dataset, plot_true_sine, generate_polynomial_features, calculate_rmse

output_dir = "output part 2"
os.makedirs(output_dir, exist_ok=True)

def run_part2():
    # Load datasets
    x_train, y_train = load_dataset('A1 - Data Sets/poly_train.csv')
    x_test, y_test = load_dataset('A1 - Data Sets/poly_test.csv')
    
    d = 9
    Phi_train = generate_polynomial_features(x_train, d)
    Phi_test = generate_polynomial_features(x_test, d)
    
    lams = np.logspace(-10, 2, 49)
    
    # ---------------------------------------------------------
    # Deliverable 1: Closed-form fits for all lambdas
    # ---------------------------------------------------------
    I = np.eye(d + 1)
    w_closed_all = []
    train_rmses = []
    test_rmses = []
    
    for lam in lams:
        # Solve (Phi^T Phi + lam * I)w = Phi^T y directly without explicit inverse
        A = Phi_train.T @ Phi_train + lam * I
        b = Phi_train.T @ y_train
        w = np.linalg.solve(A, b)
        w_closed_all.append(w)
        
        train_rmses.append(calculate_rmse(y_train, Phi_train @ w))
        test_rmses.append(calculate_rmse(y_test, Phi_test @ w))
        
    w_closed_all = np.array(w_closed_all)
    
    # ---------------------------------------------------------
    # Deliverable 2: GD Agreement at lambda = 10^-2
    # ---------------------------------------------------------
    lam_gd = 10**-2
    eta = 0.04
    
    A_gd = Phi_train.T @ Phi_train + lam_gd * I
    b_gd = Phi_train.T @ y_train
    w_closed_gd = np.linalg.solve(A_gd, b_gd)
    norm_w_closed_gd = np.linalg.norm(w_closed_gd)
    
    w_gd = np.zeros(d + 1)
    losses_gd = []
    iters = 0
    rel_error = 1.0
    
    while rel_error >= 1e-6:
        # Objective: 0.5 * ||Phi w - y||^2 + (lam/2) * ||w||^2
        current_loss = 0.5 * np.sum((Phi_train @ w_gd - y_train)**2) + (lam_gd / 2) * np.sum(w_gd**2)
        losses_gd.append(current_loss)
        
        # Gradient: Phi^T(Phi w - y) + lam * w
        grad = Phi_train.T @ (Phi_train @ w_gd - y_train) + lam_gd * w_gd
        w_gd = w_gd - eta * grad
        
        rel_error = np.linalg.norm(w_gd - w_closed_gd) / norm_w_closed_gd
        iters += 1

    # Final loss for the last iteration
    losses_gd.append(0.5 * np.sum((Phi_train @ w_gd - y_train)**2) + (lam_gd / 2) * np.sum(w_gd**2))
    
    plt.figure(figsize=(8, 6))
    plt.plot(losses_gd, color='purple', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel(r'Ridge Objective $\tilde{E}(w)$')
    plt.title(r'Deliverable 2: GD Convergence at $\lambda = 10^{-2}$')
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(output_dir, 'deliverable_2_gd_convergence.png'))
    plt.close()

    cond_unpenalized = np.linalg.cond(Phi_train.T @ Phi_train)
    cond_penalized = np.linalg.cond(A_gd)

    # ---------------------------------------------------------
    # Deliverable 3: RMSE Sweep & Selecting lambda*
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(lams, train_rmses, label='Train RMSE', marker='o', markersize=4)
    plt.plot(lams, test_rmses, label='Test RMSE', marker='s', markersize=4)
    plt.axhline(y=0.1, color='black', linestyle='--', label='Noise Level (0.1)')
    plt.xscale('log')
    plt.xlabel(r'Regularization Parameter $\lambda$ (Log Scale)')
    plt.ylabel('RMSE')
    plt.title('Deliverable 3: RMSE vs. $\lambda$')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, 'deliverable_3_rmse_sweep.png'))
    plt.close()

    # Find optimal lambda
    opt_idx = np.argmin(test_rmses)
    lam_star = lams[opt_idx]
    min_test_rmse = test_rmses[opt_idx]
    
    # Interval where Test RMSE <= 1.05 * minimum
    threshold = 1.05 * min_test_rmse
    valid_indices = np.where(np.array(test_rmses) <= threshold)[0]
    lam_interval = (lams[valid_indices[0]], lams[valid_indices[-1]])

    # ---------------------------------------------------------
    # Deliverable 4: Fitted Curves
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Deliverable 4: Effect of $\lambda$ on Model Fit", fontsize=14)
    
    x_plot = np.linspace(0, 1, 200)
    Phi_plot = generate_polynomial_features(x_plot, d)
    
    target_lams = [1e-10, lam_star, 1e-2]
    titles = ["Overfit ($\\lambda=10^{-10}$)", f"Optimal ($\\lambda^*={lam_star:.2e}$)", "Underfit ($\\lambda=10^{-2}$)"]
    w_norms = []
    
    for ax, t_lam, title in zip(axes, target_lams, titles):
        # Retrieve the pre-computed closed-form weights for this lambda
        idx = np.abs(lams - t_lam).argmin() 
        w = w_closed_all[idx]
        w_norms.append(np.linalg.norm(w))
        
        y_pred_plot = Phi_plot @ w
        ax.scatter(x_train, y_train, color='red', label='Train Data')
        plot_true_sine(ax)
        ax.plot(x_plot, y_pred_plot, color='blue', label='Ridge Fit')
        ax.set_title(title)
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_4_fitted_curves.png'))
    plt.close()

    # ---------------------------------------------------------
    # Generate Written Report
    # ---------------------------------------------------------
    with open(os.path.join(output_dir, 'part2_report.txt'), 'w') as f:
        f.write("PART 2 REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("Deliverable 2: GD Agreement\n")
        f.write("-" * 30 + "\n")
        f.write(f"Iterations to reach relative error < 1e-6: {iters}\n")
        f.write(f"cond(Phi^T Phi):               {cond_unpenalized:.2e}\n")
        f.write(f"cond(Phi^T Phi + lambda*I):    {cond_penalized:.2e}\n")
        f.write("Conditioning Discussion: The addition of the lambda penalty dramatically decreased the condition number, stabilizing the Hessian matrix and making standard gradient descent practical without requiring QR reparameterization.\n\n")
        
        f.write("Deliverable 3: RMSE Sweep\n")
        f.write("-" * 30 + "\n")
        f.write(f"Optimal lambda (lambda*):      {lam_star:.4e}\n")
        f.write(f"Test RMSE at lambda*:          {min_test_rmse:.6f}\n")
        f.write(f"Interval (<= 1.05 * min RMSE): [{lam_interval[0]:.4e}, {lam_interval[1]:.4e}]\n")
        f.write("Interval Discussion: The minimum is relatively flat within the threshold interval. To the left of the interval (lower lambda), the model overfits, causing test RMSE to spike while train RMSE stays low. To the right of the interval (higher lambda), the model underfits, causing both train and test RMSE to increase.\n\n")

        f.write("Deliverable 4: Fitted Curves Norms\n")
        f.write("-" * 30 + "\n")
        f.write(f"||w||_2 at lambda = 1e-10:     {w_norms[0]:.4f}\n")
        f.write(f"||w||_2 at lambda = lambda*:   {w_norms[1]:.4f}\n")
        f.write(f"||w||_2 at lambda = 1e-2:      {w_norms[2]:.4f}\n")

if __name__ == "__main__":
    run_part2()
    print(f"Part 2 complete. Check the '{output_dir}' directory for outputs.")