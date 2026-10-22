import os
import numpy as np
import matplotlib.pyplot as plt
from common import load_dataset, plot_true_sine, generate_polynomial_features

# Create output directory
output_dir = "output part 1"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility (required for shuffling)
SEED = 42
rng = np.random.default_rng(SEED)

def calculate_loss(Q: np.ndarray, alpha: np.ndarray, y: np.ndarray) -> float:
    """Computes the Sum of Squared Errors (SSE) loss E(alpha) = 0.5 * ||Q * alpha - y||^2"""
    return 0.5 * np.sum((Q @ alpha - y) ** 2)

def run_part1():
    # Load dataset
    x_train, y_train = load_dataset('A1 - Data Sets/poly_train.csv')
    N = len(x_train)

    # ---------------------------------------------------------
    # Deliverable 1: Fit polynomials of degree d=1...9
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 10))
    plt.suptitle("Deliverable 1: Closed-Form Polynomial Fits (d=1 to 9)")
    
    x_plot = np.linspace(0, 1, 200)
    
    for d in range(1, 10):
        Phi = generate_polynomial_features(x_train, d)
        # Robust SVD-based solver for closed-form
        w_closed, _, _, _ = np.linalg.lstsq(Phi, y_train, rcond=None)
        
        Phi_plot = generate_polynomial_features(x_plot, d)
        y_pred_plot = Phi_plot @ w_closed
        
        plt.subplot(3, 3, d)
        plt.scatter(x_train, y_train, color='red', label='Train Data')
        plot_true_sine(plt.gca())
        plt.plot(x_plot, y_pred_plot, color='blue', label=f'Fit d={d}')
        plt.title(f'Degree {d}')
        plt.ylim(-1.5, 1.5)
        if d == 1:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_1_fits.png'))
    plt.close()

    # ---------------------------------------------------------
    # Deliverable 2 & 3: GD Implementation & Loss Curves (d=9)
    # ---------------------------------------------------------
    d = 9
    Phi = generate_polynomial_features(x_train, d)
    
    # Save the closed-form weights for later verification (Deliverable 5)
    w_closed, _, _, _ = np.linalg.lstsq(Phi, y_train, rcond=None)
    
    # Thin QR Decomposition for Numerical Stability
    Q, R = np.linalg.qr(Phi)
    
    epochs = 2000
    eta = 0.1
    
    # Initialize alphas to zero
    alpha_batch = np.zeros(d + 1)
    alpha_sgd = np.zeros(d + 1)
    alpha_mb = np.zeros(d + 1)
    
    losses_batch = []
    losses_sgd = []
    losses_mb = []
    
    for epoch in range(epochs):
        # --- Batch GD ---
        # Full sum over all N points (do not divide by N)
        grad_batch = Q.T @ (Q @ alpha_batch - y_train)
        alpha_batch = alpha_batch - eta * grad_batch
        losses_batch.append(calculate_loss(Q, alpha_batch, y_train))
        
        # Shuffle for SGD and Mini-batch
        indices = rng.permutation(N)
        
        # --- Stochastic GD (SGD) ---
        for i in indices:
            Q_i = Q[i:i+1, :]
            y_i = y_train[i:i+1]
            grad_sgd = Q_i.T @ (Q_i @ alpha_sgd - y_i) # Contribution of single point
            alpha_sgd = alpha_sgd - eta * grad_sgd
        losses_sgd.append(calculate_loss(Q, alpha_sgd, y_train))
        
        # --- Mini-batch GD ---
        batch_size = 3
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_indices = indices[start_idx:end_idx]
            Q_b = Q[batch_indices, :]
            y_b = y_train[batch_indices]
            
            # Averaged gradient over the mini-batch
            grad_mb = (Q_b.T @ (Q_b @ alpha_mb - y_b)) / len(y_b)
            alpha_mb = alpha_mb - eta * grad_mb
        losses_mb.append(calculate_loss(Q, alpha_mb, y_train))

    # Plot Deliverable 3
    plt.figure(figsize=(8, 6))
    plt.plot(losses_batch, label='Batch GD', linewidth=4, alpha=0.6)
    plt.plot(losses_sgd, label='Stochastic GD', linestyle='--', linewidth=2)
    plt.plot(losses_mb, label='Mini-batch GD (size 3)', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Objective E(w) - Log Scale')
    plt.title(r'Deliverable 3: Loss Curves for GD Variants ($\eta=0.1$)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, 'deliverable_3_loss_curves.png'))
    plt.close()

    # ---------------------------------------------------------
    # Deliverable 4: Learning-rate study (Batch GD)
    # ---------------------------------------------------------
    etas = [0.01, 0.67, 2.1]
    epochs_study = 100
    
    plt.figure(figsize=(8, 6))
    
    with open(os.path.join(output_dir, 'deliverable_4_report.txt'), 'w') as f:
        f.write("Deliverable 4: Learning Rate Study Report\n")
        f.write("="*40 + "\n")
        
        for current_eta in etas:
            alpha_test = np.zeros(d + 1)
            losses_test = []
            for _ in range(epochs_study):
                grad = Q.T @ (Q @ alpha_test - y_train)
                alpha_test = alpha_test - current_eta * grad
                losses_test.append(calculate_loss(Q, alpha_test, y_train))
            
            plt.plot(losses_test, label=f'$\eta={current_eta}$')
            
            if current_eta > 2.0:
                # Recover w for the divergent case
                # Note: alpha might be infinite/NaN here, causing w to be NaN
                w_divergent = np.linalg.inv(R) @ alpha_test
                f.write(f"\nRegime: Divergence (eta={current_eta})\n")
                f.write(f"Objective E after {epochs_study} epochs: {losses_test[-1]}\n")
                f.write(f"Weight vector w:\n{w_divergent}\n")
                f.write("At exactly eta=2, the weights oscillate indefinitely because the step size perfectly overshoots the minimum by the same distance every time.\n")

    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Objective E(w) - Log Scale')
    plt.title('Deliverable 4: Batch GD Learning Rate Study')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, 'deliverable_4_lr_study.png'))
    plt.close()

    # ---------------------------------------------------------
    # Deliverable 5: Verification
    # ---------------------------------------------------------
    # Recover original space weights using w = R^-1 * alpha
    R_inv = np.linalg.inv(R)
    w_batch = R_inv @ alpha_batch
    w_sgd = R_inv @ alpha_sgd
    w_mb = R_inv @ alpha_mb
    
    norm_w_closed = np.linalg.norm(w_closed)
    
    with open(os.path.join(output_dir, 'deliverable_5_verification.txt'), 'w') as f:
        f.write("Deliverable 5: Verification against Closed-Form (d=9)\n")
        f.write("="*60 + "\n")
        
        for name, w_gd in zip(['Batch GD', 'Stochastic GD', 'Mini-batch GD'], [w_batch, w_sgd, w_mb]):
            abs_err = np.linalg.norm(w_gd - w_closed)
            rel_err = abs_err / norm_w_closed
            f.write(f"{name}:\n")
            f.write(f"  Absolute Error ||w_GD - w_closed||_2: {abs_err:.6e}\n")
            f.write(f"  Relative Error:                       {rel_err:.6e}\n\n")

if __name__ == "__main__":
    run_part1()
    print(f"Part 1 complete. Check the '{output_dir}' directory for outputs.")