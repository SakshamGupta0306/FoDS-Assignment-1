"""
Problem 3: Effect of Preprocessing, and the Two-Dimensional Projection
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_data,
    compute_covariance_matrix,
    eigendecomposition_pca,
    explained_variance_ratio,
    FEATURE_NAMES,
)


def run_p3():
    print("=" * 60)
    print("Running Problem 3: Effect of Preprocessing")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    train_csv = os.path.join(os.path.dirname(__file__), "wine_train.csv")
    test_csv = os.path.join(os.path.dirname(__file__), "wine_test.csv")
    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)

    N_train = X_train.shape[0]

    # Preprocessing Variants (C1: training statistics only)
    # V1: Centred only (subtract training mean, no scaling)
    mu_train = np.mean(X_train, axis=0)
    X_train_v1 = X_train - mu_train
    X_test_v1 = X_test - mu_train

    # Raw sample variances for V1 (divisor N - 1)
    raw_sample_var = np.var(X_train, axis=0, ddof=1)
    top3_var_idx = np.argsort(raw_sample_var)[::-1][:3]

    print("\n--- V1: Top 3 Features with Largest Raw Sample Variance ---")
    for idx in top3_var_idx:
        print(f"  - {FEATURE_NAMES[idx]}: raw variance = {raw_sample_var[idx]:.4f}")

    # V2: Min-Max
    # x'_nj = (x_nj - min_j) / (max_j - min_j) using training min and max,
    # followed by subtracting training mean of the scaled features.
    min_train = np.min(X_train, axis=0)
    max_train = np.max(X_train, axis=0)
    denom = max_train - min_train

    X_train_scaled = (X_train - min_train) / denom
    X_test_scaled = (X_test - min_train) / denom

    mu_v2 = np.mean(X_train_scaled, axis=0)
    X_train_v2 = X_train_scaled - mu_v2
    X_test_v2 = X_test_scaled - mu_v2

    # V3: Z-score (C2)
    s_train = np.std(X_train, axis=0, ddof=1)
    X_train_v3 = (X_train - mu_train) / s_train
    X_test_v3 = (X_test - mu_train) / s_train

    variants = [
        ("V1 (Centred only)", X_train_v1, X_test_v1),
        ("V2 (Min-Max)", X_train_v2, X_test_v2),
        ("V3 (Z-score)", X_train_v3, X_test_v3),
    ]

    table_rows = []
    projections = {}

    for name, X_tr, X_te in variants:
        # Sample covariance matrix S
        S = compute_covariance_matrix(X_tr)
        lambdas, U = eigendecomposition_pca(S)

        r, R = explained_variance_ratio(lambdas)
        r1, r2 = r[0], r[1]
        r1_plus_r2 = r1 + r2

        # Feature with largest |loading| on PC1
        top_pc1_idx = np.argmax(np.abs(U[:, 0]))
        top_pc1_feature = FEATURE_NAMES[top_pc1_idx]

        # 2-D Projections: cn = U_2.T @ x'_n
        U2 = U[:, :2]  # (D, 2)
        proj_train = X_tr @ U2
        proj_test = X_te @ U2

        # 5-NN classifier
        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        knn.fit(proj_train, y_train)
        acc_test = knn.score(proj_test, y_test)

        table_rows.append(
            {
                "Variant": name.split()[0],
                "r1": r1,
                "r2": r2,
                "r1 + r2": r1_plus_r2,
                "Top PC1 Feature": top_pc1_feature,
                "5-NN Test Acc": acc_test,
            }
        )

        projections[name.split()[0]] = (proj_train, proj_test)

    table_df = pd.DataFrame(table_rows)
    print("\n--- Deliverable 1: Preprocessing Comparison Table ---")
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Deliverable 3: One figure with three panels (V1, V2, V3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_titles = {
        "V1": "V1: Centred Only",
        "V2": "V2: Min-Max Scaled",
        "V3": "V3: Z-score Standardised",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    class_labels = ["Class 0", "Class 1", "Class 2"]

    for ax, v_key in zip(axes, ["V1", "V2", "V3"]):
        tr_proj, te_proj = projections[v_key]

        for c_idx in range(3):
            # Training points: filled circles
            mask_tr = (y_train == c_idx)
            ax.scatter(
                tr_proj[mask_tr, 0],
                tr_proj[mask_tr, 1],
                color=colors[c_idx],
                marker="o",
                alpha=0.75,
                edgecolors="none",
                s=40,
                label=f"Train {class_labels[c_idx]}",
            )

            # Test points: crosses
            mask_te = (y_test == c_idx)
            ax.scatter(
                te_proj[mask_te, 0],
                te_proj[mask_te, 1],
                color=colors[c_idx],
                marker="x",
                s=55,
                linewidths=1.8,
                label=f"Test {class_labels[c_idx]}",
            )

        ax.set_title(panel_titles[v_key], fontsize=13, fontweight="bold")
        ax.set_xlabel("Principal Component 1 (PC1)")
        ax.set_ylabel("Principal Component 2 (PC2)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "p3_projections_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved 3-panel projection plot to: {plot_path}")

    return {
        "table_df": table_df,
        "top3_var": [(FEATURE_NAMES[i], raw_sample_var[i]) for i in top3_var_idx],
    }


if __name__ == "__main__":
    run_p3()

