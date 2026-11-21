"""
Problem 1: PCA from First Principles
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Add current dir to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_data,
    compute_standardisation_params,
    standardise,
    compute_covariance_matrix,
    eigendecomposition_pca,
    FEATURE_NAMES,
)


def run_p1():
    print("=" * 60)
    print("Running Problem 1: PCA from First Principles")
    print("=" * 60)

    # 1. Load data
    train_csv = os.path.join(os.path.dirname(__file__), "wine_train.csv")
    test_csv = os.path.join(os.path.dirname(__file__), "wine_test.csv")
    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)

    # 2. Standardisation C1, C2
    mu, s = compute_standardisation_params(X_train)
    Z_train = standardise(X_train, mu, s)

    # 3. Compute sample covariance matrix S (C3)
    S = compute_covariance_matrix(Z_train)

    # 4. Eigendecomposition from scratch (C4)
    lambdas, U = eigendecomposition_pca(S)

    # Deliverable 2: Print the 13 eigenvalues in descending order in exact format
    print("\n--- Eigenvalues (4 decimal places) ---")
    for i, lam in enumerate(lambdas, 1):
        print(f"lambda {i} = {lam:.4f}")

    # Deliverable 3: Report ||U.T @ U - I||_F, tr(S) - sum(lambda_j), and tr(S)
    I_D = np.eye(len(lambdas))
    ortho_residual = np.linalg.norm(U.T @ U - I_D, ord="fro")
    trace_S = np.trace(S)
    trace_residual = trace_S - np.sum(lambdas)

    print("\n--- Matrix Checks ---")
    print(f"||U^T U - I||_F = {ortho_residual:.4e}")
    print(f"tr(S) = {trace_S:.4f}")
    print(f"tr(S) - sum(lambda_j) = {trace_residual:.4e}")

    # Deliverable 4: Verification against sklearn.decomposition.PCA
    pca = PCA(n_components=13, svd_solver="full")
    pca.fit(Z_train)

    # Apply sign flip rule to components_ (each row is an eigenvector)
    sk_components = pca.components_.copy()
    for row in range(sk_components.shape[0]):
        c = sk_components[row, :]
        max_abs_idx = np.argmax(np.abs(c))
        if c[max_abs_idx] < 0:
            sk_components[row, :] = -c

    sk_U = sk_components.T
    sk_lambdas = pca.explained_variance_

    diff_lambdas = np.max(np.abs(lambdas - sk_lambdas))
    diff_U = np.max(np.abs(U - sk_U))

    print("\n--- Verification with scikit-learn ---")
    print(f"Max absolute difference in eigenvalues: {diff_lambdas:.4e}")
    print(f"Max absolute difference in eigenvectors (U): {diff_U:.4e}")
    print(f"Eigenvalues agree (< 1e-8): {diff_lambdas < 1e-8}")
    print(f"Eigenvectors agree (< 1e-8): {diff_U < 1e-8}")

    # Deliverable 5: Table of loadings for u1 and u2 by feature name
    loadings_df = pd.DataFrame(
        {
            "Feature Name": FEATURE_NAMES,
            "PC1 Loading (u1)": U[:, 0],
            "PC2 Loading (u2)": U[:, 1],
            "|u1|": np.abs(U[:, 0]),
            "|u2|": np.abs(U[:, 1]),
        }
    )

    print("\n--- Loadings for PC1 and PC2 ---")
    print(
        loadings_df[["Feature Name", "PC1 Loading (u1)", "PC2 Loading (u2)"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    top3_pc1_indices = np.argsort(np.abs(U[:, 0]))[::-1][:3]
    top3_pc2_indices = np.argsort(np.abs(U[:, 1]))[::-1][:3]

    print("\nTop 3 features with largest |loading| on PC1:")
    for idx in top3_pc1_indices:
        print(f"  - {FEATURE_NAMES[idx]}: loading = {U[idx, 0]:.4f} (|loading| = {abs(U[idx, 0]):.4f})")

    print("\nTop 3 features with largest |loading| on PC2:")
    for idx in top3_pc2_indices:
        print(f"  - {FEATURE_NAMES[idx]}: loading = {U[idx, 1]:.4f} (|loading| = {abs(U[idx, 1]):.4f})")

    return {
        "lambdas": lambdas,
        "U": U,
        "ortho_residual": ortho_residual,
        "trace_S": trace_S,
        "trace_residual": trace_residual,
        "diff_lambdas": diff_lambdas,
        "diff_U": diff_U,
        "loadings_df": loadings_df,
        "top3_pc1": [FEATURE_NAMES[i] for i in top3_pc1_indices],
        "top3_pc2": [FEATURE_NAMES[i] for i in top3_pc2_indices],
    }


if __name__ == "__main__":
    run_p1()

