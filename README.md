# Foundations of Data Science: Assignment 1

Regression techniques implemented from scratch in Python (NumPy): polynomial regression, ridge, lasso and Bayesian linear regression.

## Parts

| Part | Topic | Script | Output folder |
|------|-------|--------|---------------|
| P1 | Polynomial regression: closed form (SVD/`lstsq`) vs. Batch, Stochastic and Mini-batch GD in a QR-reparametrised basis | `p1_polynomial_regression.py` | `output part 1/` |
| P2 | Ridge regression as a MAP estimate: closed form, GD in the original basis, lambda sweep | `p2_ridge_regression.py` | `output part 2/` |
| P3 | Lasso via subgradient descent in a shifted Legendre basis; Lasso vs. Ridge | `p3_L1_regularised_regression.py` | `output part 3/` |
| P4 | Bayesian linear regression: MAP/ridge equivalence and effect of N on predictive uncertainty | `p4_bayesian_regression.py` | `output part 4/` |

## Project structure

```
.
├── A1 - Data Sets/
│   ├── poly_train.csv
│   ├── poly_test.csv
│   ├── p4_train_N5.csv
│   ├── p4_train_N10.csv
│   ├── p4_train_N20.csv
│   └── p4_train_N50.csv
├── common.py                         # data loading, RMSE, polynomial features, report writer
├── p1_polynomial_regression.py
├── p2_ridge_regression.py
├── p3_L1_regularised_regression.py
├── p4_bayesian_regression.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt # numpy, pandas, matplotlib
```

## Running

Run from the project root (the folder containing `A1 - Data Sets/`). Each script creates its own output folder with the plots (`.png`) and a numerical report (`.txt`).

```bash
python3 p1_polynomial_regression.py
python3 p2_ridge_regression.py
python3 p3_L1_regularised_regression.py
python3 p4_bayesian_regression.py
```

P3 runs 20,000 subgradient iterations for each of 21 lambdas, so it takes a little longer than the others.

## Implementation notes

- **Data:** the provided CSVs are loaded as-is. sin(2πx) is used for plotting only, never for fitting.
- **No explicit inverses:** closed-form solutions use `np.linalg.lstsq` or `np.linalg.solve`.
- **P1:** the objective is the unnormalised SSE, E = ½‖Φw − t‖². GD runs on thin QR, Φ = QR, with α = Rw, starting from α = 0, and w is recovered with `solve(R, α)`. SGD uses single-row updates (10 per epoch) and Mini-batch uses batches of 3, 3, 3, 1 with the averaged gradient. Both reshuffle each epoch with `np.random.default_rng(42)` (seed = 42).
- **P2:** plain batch GD in the monomial basis with η = 0.04 from w = 0. Grid: `np.logspace(-10, 2, 49)`.
- **P3:** η = 0.05, w₀ = 0, 20,000 iterations, grid `np.logspace(-4, 0, 21)`. A coefficient counts as ≈ 0 if |wⱼ| < 2ηλ. Best-so-far E is used only for the convergence plot; every other reported value comes from the final iterate. Ridge in the Legendre basis is solved with `solve`.
- **P4:** α = 0.05, β = 100, degree-3 basis, with the same α and β for every N.

## Key results

| Item | Result |
|------|--------|
| P1: cond(Φ), d = 9 | 1.5e7 |
| P1: relative error vs. closed form (all three GD variants, 2000 epochs) | about 1.8e-11 |
| P2: GD iterations to reach 1e-6 relative error (λ = 1e-2) | 28,227 |
| P2: cond(ΦᵀΦ) → cond(ΦᵀΦ + λI) | 2.3e14 → 2.0e3 |
| P2: λ*, test RMSE | 1.78e-7, 0.1146 |
| P3: Lasso λ*, test RMSE, ≈ 0 coefficients | 0.1, 0.0985, indices [0, 6, 8, 9] |
| P3: ‖w − w*‖₂ at λ = 0.1 | 0.0076 |
| P3: Ridge best λ, test RMSE | 0.1585, 0.1422 |
| P4: ‖w_MAP − m_N‖₂ | 1.1e-12 |

Full numbers and discussions are in each `output part N/partN_report.txt`.

## Outputs

- **P1:** closed-form fits (d = 1..9), GD loss curves, learning-rate study.
- **P2:** GD convergence, train/test RMSE vs. λ, fitted curves (overfit, λ*, underfit).
- **P3:** convergence at λ = 0.1, RMSE sweep, coefficient path, Lasso vs. Ridge fit.
- **P4:** predictive distribution with the MAP overlay, and the interval width vs. N with the noise floor.