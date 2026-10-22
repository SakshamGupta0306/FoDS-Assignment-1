import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the dataset from a CSV file.
    The assignment specifies that all files have fixed columns 'x' and 'y', 
    which must be loaded directly without generation or modification[cite: 1].
    """
    df = pd.read_csv(filepath)
    # Ensure vectors are returned as numpy arrays for matrix operations
    return df['x'].values, df['y'].values

def plot_true_sine(ax: plt.Axes, num_points: int = 200) -> None:
    """
    Plots the true function sin(2πx). 
    This function is known and may be used for plotting only, never for fitting[cite: 1].
    """
    x_grid = np.linspace(0, 1, num_points)
    y_true = np.sin(2 * np.pi * x_grid)
    ax.plot(x_grid, y_true, color='black', linestyle='--', label=r'$\sin(2\pi x)$')

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.
    RMSE is the primary evaluation metric used in Parts 2 and 3[cite: 3, 4].
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def generate_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Constructs the polynomial design matrix Φ.
    The n-th row is [1, x_n, ..., x_n^d][cite: 1].
    """
    N = len(x)
    Phi = np.zeros((N, degree + 1))
    for j in range(degree + 1):
        Phi[:, j] = x ** j
    return Phi