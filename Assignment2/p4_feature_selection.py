"""
Problem 4: PCA versus Feature Selection for a Downstream Classifier
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_data,
    compute_standardisation_params,
    standardise,
    compute_covariance_matrix,
    eigendecomposition_pca,
    FEATURE_NAMES,
)


def compute_f_scores(Z_train, y_train):
    """
    Compute ANOVA F-score for each feature on training set:
        F_j = [ sum_{c=1}^C N_c (m_{jc} - m_j)^2 / (C - 1) ] /
              [ sum_{c=1}^C sum_{n in c} (z_{nj} - m_{jc})^2 / (N - C) ]
    """
    N, D = Z_train.shape
    classes = np.unique(y_train)
    C = len(classes)

    # Overall feature means (for standardized data, m_j should be ~0)
    m_j = np.mean(Z_train, axis=0)

    # Class counts and class means
    N_c = np.zeros(C)
    m_jc = np.zeros((C, D))

    for c in range(C):
        mask = (y_train == c)
        N_c[c] = np.sum(mask)
        m_jc[c] = np.mean(Z_train[mask], axis=0)

    # Between-group variability: sum_{c=1}^C N_c (m_{jc} - m_j)^2 / (C - 1)
    between_var = np.zeros(D)
    for c in range(C):
        between_var += N_c[c] * (m_jc[c] - m_j) ** 2
    between_var /= (C - 1)

    # Within-group variability: sum_{c=1}^C sum_{n in c} (z_{nj} - m_{jc})^2 / (N - C)
    within_var = np.zeros(D)
    for c in range(C):
        mask = (y_train == c)
        diff = Z_train[mask] - m_jc[c]  # (N_c, D)
        within_var += np.sum(diff ** 2, axis=0)
    within_var /= (N - C)

    F = between_var / within_var
    return F


def run_p4():
    print("=" * 60)
    print("Running Problem 4: PCA vs. Feature Selection")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data
    train_csv = os.path.join(os.path.dirname(__file__), "wine_train.csv")
    test_csv = os.path.join(os.path.dirname(__file__), "wine_test.csv")
    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)

    # 2. Standardisation (C1, C2)
    mu, s = compute_standardisation_params(X_train)
    Z_train = standardise(X_train, mu, s)
    Z_test = standardise(X_test, mu, s)

    # Common classifier and folds
    #clf_base = LogisticRegression(C=1.0, max_iter=1000)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Baseline: Train on all 13 standardised training features, score on test set
    clf_full = LogisticRegression(C=1.0, max_iter=1000)
    clf_full.fit(Z_train, y_train)
    baseline_acc = clf_full.score(Z_test, y_test)
    print(f"\n--- Baseline Test Accuracy (All 13 features): {baseline_acc:.4f} ---")

    # (A) PCA from P1
    S = compute_covariance_matrix(Z_train)
    lambdas, U = eigendecomposition_pca(S)

    # (B) Filter (F-scores)
    F_scores = compute_f_scores(Z_train, y_train)
    f_rank_idx = np.argsort(F_scores)[::-1]

    print("\n--- F-score Ranking of All 13 Features ---")
    f_ranking_df = pd.DataFrame(
        {
            "Rank": np.arange(1, len(FEATURE_NAMES) + 1),
            "Feature Name": [FEATURE_NAMES[i] for i in f_rank_idx],
            "F-score": F_scores[f_rank_idx],
        }
    )
    print(f_ranking_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Evaluate across k in {1, 2, 3, 5, 8}
    k_vals = [1, 2, 3, 5, 8]
    acc_A = []
    acc_B = []
    acc_C = []

    features_B_dict = {}
    features_C_dict = {}

    for k in k_vals:
        # (A) PCA: Uk.T @ zn
        Uk = U[:, :k]
        Z_train_pca = Z_train @ Uk
        Z_test_pca = Z_test @ Uk

        clf_pca = LogisticRegression(C=1.0, max_iter=1000)
        clf_pca.fit(Z_train_pca, y_train)
        acc_A.append(clf_pca.score(Z_test_pca, y_test))

        # (B) Filter: top k features by F-score
        top_k_f_idx = f_rank_idx[:k]
        features_B_dict[k] = [FEATURE_NAMES[i] for i in top_k_f_idx]

        Z_train_f = Z_train[:, top_k_f_idx]
        Z_test_f = Z_test[:, top_k_f_idx]

        clf_f = LogisticRegression(C=1.0, max_iter=1000)
        clf_f.fit(Z_train_f, y_train)
        acc_B.append(clf_f.score(Z_test_f, y_test))

        # (C) Wrapper: Sequential Forward Selection
        sfs = SequentialFeatureSelector(
            LogisticRegression(C=1.0, max_iter=1000),
            n_features_to_select=k,
            direction="forward",
            scoring="accuracy",
            cv=folds,
            n_jobs=1,
        )
        sfs.fit(Z_train, y_train)
        sfs_mask = sfs.get_support()
        sfs_selected_idx = np.where(sfs_mask)[0]
        features_C_dict[k] = [FEATURE_NAMES[i] for i in sfs_selected_idx]

        Z_train_sfs = Z_train[:, sfs_mask]
        Z_test_sfs = Z_test[:, sfs_mask]

        clf_sfs = LogisticRegression(C=1.0, max_iter=1000)
        clf_sfs.fit(Z_train_sfs, y_train)
        acc_C.append(clf_sfs.score(Z_test_sfs, y_test))

    # Deliverable 1: Table for k in {1, 2, 3, 5, 8}
    res_df = pd.DataFrame(
        {
            "k": k_vals,
            "(A) PCA Accuracy": acc_A,
            "(B) Filter Accuracy": acc_B,
            "(C) Wrapper Accuracy": acc_C,
        }
    )
    print("\n--- Deliverable 1: Downstream Test Accuracy Table ---")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Deliverable 3: Overlap for k=3 and k=8
    print("\n--- Feature Selection Overlap Analysis ---")
    for k in [3, 8]:
        set_B = set(features_B_dict[k])
        set_C = set(features_C_dict[k])
        shared = set_B.intersection(set_C)
        print(f"\nFor k = {k}:")
        print(f"  - (B) Filter (F-score) features: {features_B_dict[k]}")
        print(f"  - (C) Wrapper (SFS) features:    {features_C_dict[k]}")
        print(f"  - Number of shared features:     {len(shared)} / {k}")
        print(f"  - Shared features:                {sorted(list(shared))}")

    # Deliverable 2: Plot of test accuracy against k with baseline
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(k_vals, acc_A, marker="o", linewidth=2, color="blue", label="(A) PCA (Unsupervised)")
    ax.plot(k_vals, acc_B, marker="s", linewidth=2, color="green", label="(B) Filter (F-score)")
    ax.plot(k_vals, acc_C, marker="^", linewidth=2, color="crimson", label="(C) Wrapper (SFS)")
    ax.axhline(
        y=baseline_acc,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline (All 13 features: {baseline_acc:.4f})",
    )

    ax.set_xlabel("Number of Retained Features / Components $k$")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy vs. $k$ for PCA, Filter, and Wrapper Methods")
    ax.set_xticks(k_vals)
    ax.set_ylim([0.7, 1.02])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "p4_accuracy_vs_k.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved Accuracy comparison plot to: {plot_path}")

    # Smallest k within one test sample of baseline.
    # Compare integer counts of correct predictions to avoid floating-point
    # errors (e.g. 52/54 vs baseline - 1/54 differ in the last bit).
    n_test = len(y_test)
    baseline_correct = int(round(baseline_acc * n_test))
    threshold_correct = baseline_correct - 1
    print(
        f"\nThreshold for within 1 test error of baseline: "
        f">= {threshold_correct}/{n_test} correct ({threshold_correct / n_test:.4f})"
    )
    for method_name, acc_list in [("(A) PCA", acc_A), ("(B) Filter", acc_B), ("(C) Wrapper", acc_C)]:
        reaching_k = [
            k for k, acc in zip(k_vals, acc_list)
            if int(round(acc * n_test)) >= threshold_correct
        ]
        smallest_k = min(reaching_k) if reaching_k else "None"
        print(f"  - {method_name}: smallest k = {smallest_k}")

    return {
        "baseline_acc": baseline_acc,
        "results_table": res_df,
        "features_B": features_B_dict,
        "features_C": features_C_dict,
    }


if __name__ == "__main__":
    run_p4()

