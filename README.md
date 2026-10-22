Foundations of Data Science - Assignment 1
This repository contains the implementation for Assignment 1, focusing on foundational regression techniques implemented from scratch in Python.Project OverviewThe assignment is divided into four main parts:Part 1: Polynomial Regression. Implements closed-form solutions (via SVD) and Gradient Descent variants (Batch, Stochastic, Mini-batch) using QR decomposition to handle severely ill-conditioned matrices.Part 2: Ridge Regression (L2 Regularization). Explores Ridge regression as a Maximum A Posteriori (MAP) estimate. Solves for weights using both closed-form equations and gradient descent, and evaluates the effect of the regularization penalty ($\lambda$).Part 3: Lasso Regression (L1 Regularization). Uses Subgradient Descent to optimize the Lasso objective. Utilizes a shifted Legendre polynomial basis to improve matrix conditioning and demonstrates the sparsity-inducing properties of the L1 norm.Part 4: Bayesian Regression. Computes the full Bayesian posterior predictive distribution, confirming its equivalence to the MAP estimate, and visualizes how predictive uncertainty decreases as dataset size ($N$) increases.
Project Structure├── A1 - Data Sets/              # Directory containing the provided CSV datasets
                 │   ├── poly_train.csv
                 │   ├── poly_test.csv
                 │   |── p4_train_N5.csv
                 |   |── p4_train_N10.csv
                 |   |── p4_train_N20.csv
                 |   └── p4_train_N50.csv
                 ├── common.py                    # Shared utilities (data loading, RMSE calculation, true sine plotting)
                 ├── p1_polynomial_regression.py                     # Polynomial Regression & Gradient Descent
                 ├── p2_ridge_regression.py                     # Ridge Regression
                 ├── p3_L1_regularised_regression.py                     # Lasso Regression
                 ├── p4_bayesian_regression.py                     # Bayesian Regression
                 ├── requirements.txt             # Python dependencies
                 └── README.md                    # Project documentation
Setup and InstallationCreate a virtual environment:python -m venv venv
Activate the virtual environment:On Windows: .\venv\Scripts\activateOn Linux/macOS: source venv/bin/activateInstall the required dependencies:pip install -r requirements.txt
(Ensure you have numpy, matplotlib, and pandas installed).ExecutionRun each part independently from the terminal. Each script will automatically generate an output directory (e.g., output part 1/) containing all required plots (.png) and numerical reports (.txt).
python3 p1_polynomial_regression.py
python3 p2_ridge_regression.py
python3 p3_L1_regularised_regression.py
python3 p4_bayesian_regression.py