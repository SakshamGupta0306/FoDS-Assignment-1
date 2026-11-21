# CS F320: Foundations of Data Science — Assignment 2 Report
**Topic:** Principal Component Analysis & Dimensionality Reduction  
**Submission Deadline:** 2355 Hrs on 23rd Nov 2026  
**Dataset:** Wine Dataset (`wine_train.csv`: $N=124$, `wine_test.csv`: $N_{\text{test}}=54$, $D=13$)

---

## Problem 1: PCA from First Principles

### 1. Eigenvalues Output (Printed to 4 decimal places in descending order)
```text
lambda 1 = 4.6450
lambda 2 = 2.4972
lambda 3 = 1.4309
lambda 4 = 0.9426
lambda 5 = 0.9065
lambda 6 = 0.6944
lambda 7 = 0.5922
lambda 8 = 0.3140
lambda 9 = 0.2653
lambda 10 = 0.2570
lambda 11 = 0.2191
lambda 12 = 0.1413
lambda 13 = 0.0947
```

### 2. Matrix Sanity & Orthonormality Checks
- **Frobenius norm of orthonormality residual:**  
  $$\|\mathbf{U}^\top \mathbf{U} - \mathbf{I}\|_F = 3.2429\text{e-}15$$
- **Trace of covariance matrix $\mathbf{S}$:**  
  $$\operatorname{tr}(\mathbf{S}) = 13.0000$$
- **Trace identity difference:**  
  $$\operatorname{tr}(\mathbf{S}) - \sum_{j=1}^{13} \lambda_j = 1.7764\text{e-}15$$

### 3. Verification with `scikit-learn`
- **Max absolute difference in eigenvalues** ($|\lambda_j - \text{explained\_variance\_}_j|$):  
  $$4.4409\text{e-}15 \quad (< 10^{-8}, \text{Agrees})$$
- **Max absolute difference in eigenvectors** ($\max |\mathbf{U} - \text{components\_}^\top|$ after canonical sign-flip):  
  $$8.3961\text{e-}15 \quad (< 10^{-8}, \text{Agrees})$$

### 4. Loadings Table for PC1 and PC2
| Feature Name | PC1 Loading ($\mathbf{u}_1$) | PC2 Loading ($\mathbf{u}_2$) |
| :--- | :---: | :---: |
| `alcohol` | 0.1576 | 0.4941 |
| `malic_acid` | -0.2524 | 0.1671 |
| `ash` | -0.0166 | 0.3358 |
| `alcalinity_of_ash` | -0.2351 | -0.0230 |
| `magnesium` | 0.1429 | 0.2817 |
| `total_phenols` | 0.3899 | 0.0716 |
| `flavanoids` | 0.4260 | 0.0057 |
| `nonflavanoid_phenols` | -0.2868 | 0.0486 |
| `proanthocyanins` | 0.3086 | 0.0059 |
| `color_intensity` | -0.0889 | 0.5263 |
| `hue` | 0.3012 | -0.2840 |
| `od280/od315_of_diluted_wines` | 0.3778 | -0.1814 |
| `proline` | 0.2884 | 0.3703 |

- **Top 3 features with largest $|\text{loading}|$ on PC1:**
  1. `flavanoids`: $0.4260$
  2. `total_phenols`: $0.3899$
  3. `od280/od315_of_diluted_wines`: $0.3778$
- **Top 3 features with largest $|\text{loading}|$ on PC2:**
  1. `color_intensity`: $0.5263$
  2. `alcohol`: $0.4941$
  3. `proline`: $0.3703$

---

## Problem 2: Variance Captured vs. Number of Components

### 1. Dimension Retention Criteria
- **Elbow criterion ($k_{\text{elbow}} = \max\{k : r_k \ge 0.10\}$):**  
  $$k_{\text{elbow}} = 3$$  
  *(PC3 explains $11.01\%$ of variance, with cumulative variance $R(3) = 65.95\%$)*
- **Kaiser criterion ($\lambda_j > 1.0$):**  
  $$\text{Number of retained components} = 3 \quad (\lambda_1=4.6450, \lambda_2=2.4972, \lambda_3=1.4309)$$
- **Cumulative Variance Thresholds ($k_\tau = \min\{k : R(k) \ge \tau\}$):**
  - $k_{0.80} = 5 \quad (R(5) = 80.17\%)$
  - $k_{0.90} = 7 \quad (R(7) = 90.07\%)$
  - $k_{0.95} = 10 \quad (R(10) = 96.50\%)$

### 2. Summary Table for $k = 1, \dots, 13$
| $k$ | $r_k$ | $R(k)$ | $J_{\text{train}}(k)$ | $J_{\text{test}}(k)$ |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3573 | 0.3573 | 8.3550 | 8.5455 |
| 2 | 0.1921 | 0.5494 | 5.8578 | 6.1120 |
| 3 | 0.1101 | 0.6595 | 4.4270 | 4.5980 |
| 4 | 0.0725 | 0.7320 | 3.4844 | 3.8979 |
| 5 | 0.0697 | 0.8017 | 2.5779 | 3.0326 |
| 6 | 0.0534 | 0.8551 | 1.8835 | 2.4541 |
| 7 | 0.0456 | 0.9007 | 1.2913 | 1.8367 |
| 8 | 0.0242 | 0.9248 | 0.9773 | 1.3861 |
| 9 | 0.0204 | 0.9452 | 0.7120 | 1.0353 |
| 10 | 0.0198 | 0.9650 | 0.4550 | 0.7532 |
| 11 | 0.0169 | 0.9819 | 0.2359 | 0.4497 |
| 12 | 0.0109 | 0.9927 | 0.0947 | 0.1726 |
| 13 | 0.0073 | 1.0000 | 0.0000 | 0.0000 |

### 3. Numerical Verification of Reconstruction-Error Identity
$$\max_{k=1,\dots,13} \left| J_{\text{train}}(k) - \sum_{j=k+1}^{13} \lambda_j \right| = 2.6645\text{e-}15 \quad (< 10^{-8})$$

### 4. Generated Plots
- Scree & Cumulative Variance Plot: `output/p2_scree_cumulative.png`
- Reconstruction Error Plot ($J_{\text{train}}(k)$ & $J_{\text{test}}(k)$): `output/p2_reconstruction_error.png`

### 5. Discussion Paragraph
The elbow rule gives $k_{\text{elbow}} = 3$, which exactly matches the Kaiser count of 3 components ($\lambda_j > 1$), but falls noticeably short of standard cumulative variance thresholds ($k_{0.80}=5, k_{0.90}=7, k_{0.95}=10$). Because $R(k_{\text{elbow}}) = 65.95\%$, stopping at the elbow discards roughly one-third of the dataset's total variance, suggesting that $k=3$ is very aggressive for reconstruction. Algebraically, the squared reconstruction error for any point $\mathbf{z}_n$ decomposes as $\|\mathbf{z}_n - \hat{\mathbf{z}}_n\|^2 = \sum_{j>k} (\mathbf{u}_j^\top \mathbf{z}_n)^2$. Increasing $k$ to $k+1$ removes the non-negative term $(\mathbf{u}_{k+1}^\top \mathbf{z}_n)^2 \ge 0$, guaranteeing that $J_{\text{test}}(k)$ is strictly non-increasing in $k$ for any dataset. Indeed, the plot confirms that $J_{\text{test}}(k)$ monotonically decreases from 8.5455 down to 0.0000 at $k=13$, tracking slightly above $J_{\text{train}}(k)$ due to slight generalization gap on unseen test points.

---

## Problem 3: Effect of Preprocessing, and the Two-Dimensional Projection

### 1. Top 3 Features with Largest Raw Sample Variance (V1)
1. `proline`: raw variance $= 94906.7109$
2. `magnesium`: raw variance $= 232.0715$
3. `alcalinity_of_ash`: raw variance $= 11.4713$

### 2. Preprocessing Comparison Table
| Variant | $r_1$ | $r_2$ | $r_1 + r_2$ | Top PC1 Feature | 5-NN Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V1 (Centred only)** | 0.9978 | 0.0021 | 0.9998 | `proline` | 0.7222 |
| **V2 (Min-Max)** | 0.4271 | 0.1841 | 0.6113 | `flavanoids` | 0.9815 |
| **V3 (Z-score)** | 0.3573 | 0.1921 | 0.5494 | `flavanoids` | 0.9444 |

### 3. Generated Figures
- Three-Panel Projection Plot (V1, V2, V3): `output/p3_projections_comparison.png`

### 4. Discussion Paragraph
V1 yields $r_1 = 0.9978$ because `proline` has a massive raw variance of $94,906.71$, which completely dwarfs all other 12 features (the next largest is `magnesium` at $232.07$), so the first eigenvector trivially aligns almost entirely with the `proline` axis. Although V1 captures $99.98\%$ of the total unscaled variance in just two components, it yields the lowest 5-NN test accuracy ($72.22\%$) compared to V2 ($98.15\%$) and V3 ($94.44\%$). This demonstrates that unsupervised variance captured reflects measurement scale rather than discriminative class structure; maximizing raw variance merely recovers features measured in large units. For this dataset, V3 (or V2) should be used because standardising features prevents unit dominance and makes PCA equivalent to an eigendecomposition of the correlation matrix, allowing subtle yet highly informative biochemical markers like `flavanoids` to govern the low-dimensional embedding.

---

## Problem 4: PCA versus Feature Selection for a Downstream Classifier

### 1. Baseline Test Accuracy
- **Classifier:** Multinomial Logistic Regression (`LogisticRegression(C=1.0, max_iter=1000)`)
- **Baseline Test Accuracy (All 13 standardised features):**  
  $$\text{Baseline} = 0.9815 \quad (53/54 \text{ correct})$$

### 2. Full ANOVA $F$-score Ranking of All 13 Features
| Rank | Feature Name | $F$-score |
| :---: | :--- | :---: |
| 1 | `flavanoids` | 187.6858 |
| 2 | `proline` | 149.4335 |
| 3 | `od280/od315_of_diluted_wines` | 126.6191 |
| 4 | `alcohol` | 100.3464 |
| 5 | `color_intensity` | 83.9793 |
| 6 | `hue` | 72.2673 |
| 7 | `total_phenols` | 59.2558 |
| 8 | `alcalinity_of_ash` | 26.0528 |
| 9 | `malic_acid` | 22.5828 |
| 10 | `proanthocyanins` | 19.9799 |
| 11 | `nonflavanoid_phenols` | 17.9053 |
| 12 | `ash` | 11.5760 |
| 13 | `magnesium` | 7.1124 |

### 3. Downstream Classifier Accuracy Table
| $k$ | (A) PCA Accuracy | (B) Filter Accuracy | (C) Wrapper Accuracy |
| :---: | :---: | :---: | :---: |
| 1 | 0.8704 | 0.8519 | 0.8519 |
| 2 | 0.9444 | 0.9074 | 0.8889 |
| 3 | 0.9815 | 0.9074 | 0.8704 |
| 5 | 1.0000 | 0.9815 | 0.9074 |
| 8 | 0.9815 | 0.9815 | 0.9630 |

### 4. Feature Selection Overlap Analysis ($k = 3$ and $k = 8$)
- **For $k = 3$:**
  - (B) Filter features: `['flavanoids', 'proline', 'od280/od315_of_diluted_wines']`
  - (C) Wrapper features: `['alcohol', 'alcalinity_of_ash', 'flavanoids']`
  - Shared features: **1 / 3** (`flavanoids`)
- **For $k = 8$:**
  - (B) Filter features: `['flavanoids', 'proline', 'od280/od315_of_diluted_wines', 'alcohol', 'color_intensity', 'hue', 'total_phenols', 'alcalinity_of_ash']`
  - (C) Wrapper features: `['alcohol', 'ash', 'alcalinity_of_ash', 'magnesium', 'flavanoids', 'proanthocyanins', 'hue', 'proline']`
  - Shared features: **5 / 8** (`alcalinity_of_ash`, `alcohol`, `flavanoids`, `hue`, `proline`)

### 5. Generated Figures
- Test Accuracy vs. $k$ Curve with Baseline: `output/p4_accuracy_vs_k.png`

### 6. Discussion Paragraph
The baseline test accuracy is $0.9815$ (53/54 correct), and reaching within one test sample requires an accuracy of at least $0.9630$ (52/54 correct). PCA reaches this threshold earliest at $k=3$ (accuracy $0.9815$), Filter reaches it at $k=5$ ($0.9815$), while Wrapper reaches it only at $k=8$ ($0.9630$). Although PCA is completely unsupervised, its orthogonal linear combinations compress shared variance across multiple correlated discriminative features, allowing it to outperform single individual feature subsets at small $k$. However, because the test set contains only $N_{\text{test}} = 54$ samples, a single sample change represents a large shift of $1/54 \approx 1.85\%$, meaning that differences between $0.9630$ and $0.9815$ correspond to just one misclassified point. Consequently, one cannot be statistically confident in strictly ranking these methods based solely on such small differences without larger test samples or repeated cross-validation testing.

