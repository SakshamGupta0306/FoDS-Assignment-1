"""
Problem 2: Variance Captured vs. Number of Components
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_data,
    compute_standardisation_params,
    standardise,
    compute_covariance_matrix,
    eigendecomposition_pca,
    explained_variance_ratio,
)


def run_p2():
    print("=" * 60)
    print("Running Problem 2: Variance Captured vs. Number of Components")
    print("=" * 60)

    # Output directory for plots
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    train_csv = os.path.join(os.path.dirname(__file__), "wine_train.csv")
    test_csv = os.path.join(os.path.dirname(__file__), "wine_test.csv")
    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)

    # 2. Standardisation C1, C2
    mu, s = compute_standardisation_params(X_train)
    Z_train = standardise(X_train, mu, s)
    Z_test = standardise(X_test, mu, s)

    # 3. PCA from P1
    S = compute_covariance_matrix(Z_train)
    lambdas, U = eigendecomposition_pca(S)
    D = len(lambdas)

    # 4. Explained variance ratios
    r, R = explained_variance_ratio(lambdas)

    # Criteria calculations
    # Elbow criterion: max{k : r_k >= 0.10}
    # Note: 1-indexed k
    elbow_candidates = [k for k in range(1, D + 1) if r[k - 1] >= 0.10]
    k_elbow = max(elbow_candidates)

    # Cumulative variance thresholds: min{k : R(k) >= tau}
    k_80 = min([k for k in range(1, D + 1) if R[k - 1] >= 0.80])
    k_90 = min([k for k in range(1, D + 1) if R[k - 1] >= 0.90])
    k_95 = min([k for k in range(1, D + 1) if R[k - 1] >= 0.95])

    # Kaiser criterion: lambda_j > tr(S)/D = 1.0
    kaiser_count = sum(lambdas > (np.trace(S) / D))

    print(f"Elbow criterion (r_k >= 0.10): k_elbow = {k_elbow} (explains {r[k_elbow-1]*100:.2f}% of var, cumulative {R[k_elbow-1]*100:.2f}%)")
    print(f"Cumulative threshold tau=0.80: k_0.80 = {k_80}")
    print(f"Cumulative threshold tau=0.90: k_0.90 = {k_90}")
    print(f"Cumulative threshold tau=0.95: k_0.95 = {k_95}")
    print(f"Kaiser criterion count (lambda > 1.0): {kaiser_count}")

    # 5. Compute J_train(k) and J_test(k) for k = 1 ... 13
    N_train = Z_train.shape[0]
    N_test = Z_test.shape[0]

    J_train = []
    J_test = []

    for k in range(1, D + 1):
        Uk = U[:, :k]  # (D, k)

        # Projections and reconstructions
        Z_train_hat = Z_train @ Uk @ Uk.T
        Z_test_hat = Z_test @ Uk @ Uk.T

        # Reconstruction error: 1 / (n_D - 1) * sum(||z_n - z_hat_n||^2)
        err_train = np.sum((Z_train - Z_train_hat) ** 2) / (N_train - 1)
        err_test = np.sum((Z_test - Z_test_hat) ** 2) / (N_test - 1)

        J_train.append(err_train)
        J_test.append(err_test)

    J_train = np.array(J_train)
    J_test = np.array(J_test)

    # Deliverable: Table for k = 1 ... 13 with columns r_k, R(k), J_train(k), J_test(k)
    table_df = pd.DataFrame(
        {
            "k": np.arange(1, D + 1),
            "r_k": r,
            "R(k)": R,
            "J_train(k)": J_train,
            "J_test(k)": J_test,
        }
    )
    print("\n--- Summary Table for k = 1 ... 13 ---")
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Numerical verification of reconstruction identity:
    # J_train(k) == sum_{j=k+1}^D lambda_j
    theoretical_J_train = np.array([np.sum(lambdas[k:]) for k in range(1, D + 1)])
    diff_J_train = np.abs(J_train - theoretical_J_train)
    max_diff_identity = np.max(diff_J_train)

    print("\n--- Numerical Verification of Identity ---")
    print(f"max_{{k=1..13}} |J_train(k) - sum_{{j>k}} lambda_j| = {max_diff_identity:.4e}")
    print(f"Identity holds (< 1e-8): {max_diff_identity < 1e-8}")

    # Plot 1: Scree plot and Cumulative variance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Scree Plot
    bars = ax1.bar(np.arange(1, D + 1), r, color="steelblue", edgecolor="black", label="$r_j$ (Explained Variance Ratio)")
    ax1.axvline(x=k_elbow, color="firebrick", linestyle="--", linewidth=1.8, label=f"Elbow $k_{{elbow}} = {k_elbow}$")
    ax1.set_xlabel("Principal Component Index $j$")
    ax1.set_ylabel("Explained Variance Ratio $r_j$")
    ax1.set_title("Scree Plot")
    ax1.set_xticks(np.arange(1, D + 1))
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Cumulative Variance Plot
    ax2.plot(np.arange(1, D + 1), R, marker="o", color="darkgreen", linewidth=2, label="$R(k)$ (Cumulative Ratio)")
    ax2.axhline(y=0.80, color="gray", linestyle="--", label="80% Threshold")
    ax2.axhline(y=0.90, color="blue", linestyle="--", label="90% Threshold")
    ax2.axhline(y=0.95, color="purple", linestyle="--", label="95% Threshold")
    ax2.set_xlabel("Number of Retained Components $k$")
    ax2.set_ylabel("Cumulative Explained Variance Ratio $R(k)$")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xticks(np.arange(1, D + 1))
    ax2.set_ylim([0.3, 1.05])
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    scree_path = os.path.join(out_dir, "p2_scree_cumulative.png")
    plt.savefig(scree_path, dpi=300)
    plt.close()
    print(f"\nSaved Scree and Cumulative variance plot to: {scree_path}")

    # Plot 2: Reconstruction error J_train and J_test vs k
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, D + 1), J_train, marker="o", linewidth=2, color="royalblue", label=r"$J_{train}(k)$ (Training Error)")
    ax.plot(np.arange(1, D + 1), J_test, marker="s", linewidth=2, color="crimson", label=r"$J_{test}(k)$ (Test Error)")
    ax.set_xlabel("Number of Retained Components $k$")
    ax.set_ylabel("Reconstruction Error $J(k)$")
    ax.set_title("Reconstruction Error vs. Number of Components")
    ax.set_xticks(np.arange(1, D + 1))
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    recon_path = os.path.join(out_dir, "p2_reconstruction_error.png")
    plt.savefig(recon_path, dpi=300)
    plt.close()
    print(f"Saved Reconstruction error plot to: {recon_path}")

    return {
        "k_elbow": k_elbow,
        "k_80": k_80,
        "k_90": k_90,
        "k_95": k_95,
        "kaiser_count": kaiser_count,
        "table_df": table_df,
        "max_diff_identity": max_diff_identity,
    }


if __name__ == "__main__":
    run_p2()

