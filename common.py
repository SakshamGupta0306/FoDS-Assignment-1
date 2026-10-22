import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# Folder that holds the provided CSV files
DATA_DIR = "A1 - Data Sets"


def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a provided CSV (columns x, y) directly, without modification."""
    df = pd.read_csv(filepath)
    return df['x'].values, df['y'].values


def plot_true_sine(ax: plt.Axes, num_points: int = 200) -> None:
    """Plot sin(2*pi*x). For plotting only, never for fitting."""
    x_grid = np.linspace(0, 1, num_points)
    ax.plot(x_grid, np.sin(2 * np.pi * x_grid), color='black',
            linestyle='--', label=r'$\sin(2\pi x)$')


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE = sqrt(mean squared error)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def generate_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Monomial design matrix, n-th row = [1, x_n, ..., x_n^degree]."""
    Phi = np.zeros((len(x), degree + 1))
    for j in range(degree + 1):
        Phi[:, j] = x ** j
    return Phi


def write_report(path: str, lines) -> None:
    """Write the report to disk and echo it to the console."""
    text = "\n".join(lines) + "\n"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)