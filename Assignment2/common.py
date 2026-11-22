"""
common.py - Shared utilities and definitions for Assignment 2.
Implements C1 - C6 conventions strictly.
"""

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]


def load_data(train_path="wine_train.csv", test_path="wine_test.csv"):
    """
    Load wine_train.csv and wine_test.csv.
    Returns:
        X_train (N=124, 13), y_train (N=124,)
        X_test (N_test=54, 13), y_test (N_test=54,)
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_train = df_train["class"].to_numpy(dtype=np.int64)

    X_test = df_test[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_test = df_test["class"].to_numpy(dtype=np.int64)

    return X_train, y_train, X_test, y_test


def compute_standardisation_params(X_train):
    """
    C1 & C2: Compute training mean mu_j and training sample standard deviation s_j (divisor N-1).
    """
    #N = X_train.shape[0]
    mu = np.mean(X_train, axis=0)
    # divisor N - 1 (ddof=1)
    s = np.std(X_train, axis=0, ddof=1)
    return mu, s


def standardise(X, mu, s):
    """
    C2: z_nj = (x_nj - mu_j) / s_j
    """
    return (X - mu) / s


def compute_covariance_matrix(X_centered):
    """
    C3: S = 1/(N - 1) * X_tilde.T @ X_tilde
    """
    N = X_centered.shape[0]
    return (X_centered.T @ X_centered) / (N - 1)


def enforce_sign_rule(U):
    """
    C4: Flip each eigenvector u_j so that its entry of largest absolute value is positive.
    U has eigenvectors as columns (shape: D, D or D, k).
    """
    U_flipped = U.copy()
    for col in range(U_flipped.shape[1]):
        u = U_flipped[:, col]
        max_abs_idx = np.argmax(np.abs(u))
        if u[max_abs_idx] < 0:
            U_flipped[:, col] = -u
    return U_flipped


def eigendecomposition_pca(S):
    """
    C4: Eigendecomposition of symmetric matrix S.
    Sort eigenvalues in descending order: lambda_1 >= ... >= lambda_D.
    Returns:
        lambdas (D,): sorted eigenvalues
        U (D, D): unit-norm eigenvectors as columns with canonical sign flip.
    """
    # eigh is symmetric eigensolver
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    lambdas = eigenvalues[idx]
    U = eigenvectors[:, idx]

    # Enforce sign rule
    U = enforce_sign_rule(U)

    return lambdas, U


def explained_variance_ratio(lambdas):
    """
    C5: r_j = lambda_j / sum(lambdas), R(k) = cumsum(r)
    """
    total_var = np.sum(lambdas)
    r = lambdas / total_var
    R = np.cumsum(r)
    return r, R

