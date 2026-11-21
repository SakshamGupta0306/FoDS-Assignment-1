# CS F320: Foundations of Data Science — Assignment 2
**Topic:** Principal Component Analysis (PCA)  
**Submission Deadline:** 2355 Hrs on 23rd Nov 2026  
**Contact TA:** Mr. Bhargava (`f20230053@hyderabad.bits-pilani.ac.in`)

---

## 1. General Setup & Mandatory Constraints

### Dataset
- **Files:** `wine_train.csv` ($N = 124$ samples) and `wine_test.csv` ($N_{\text{test}} = 54$ samples).
- **Features ($D = 13$ numeric columns in exact order):**
  1. `alcohol` (col 0)
  2. `malic_acid` (col 1)
  3. `ash` (col 2)
  4. `alcalinity_of_ash` (col 3)
  5. `magnesium` (col 4)
  6. `total_phenols` (col 5)
  7. `flavanoids` (col 6)
  8. `nonflavanoid_phenols` (col 7)
  9. `proanthocyanins` (col 8)
  10. `color_intensity` (col 9)
  11. `hue` (col 10)
  12. `od280/od315_of_diluted_wines` (col 11)
  13. `proline` (col 12)
- **Target:** `class` $\in \{0, 1, 2\}$.
- **Naming Rule:** Where a deliverable asks for feature names, always report the exact column name.

### Mandatory Conventions (C1 – C6)
- **C1: Train-only Fitting:** Every statistic (mean, standard deviation, min, max, eigenvectors, feature scores) must be computed exclusively from the training set and applied unchanged to the test set. Never use the test set for fitting or selection.
- **C2: Standardisation (z-score):**
  $$z_{nj} = \frac{x_{nj} - \mu_j}{s_j}$$
  where $\mu_j$ is the training mean and $s_j$ is the sample standard deviation computed with divisor $N - 1$.
- **C3: Sample Covariance:** For mean-centred data $\tilde{\mathbf{x}}_n = \mathbf{x}_n - \bar{\mathbf{x}}$:
  $$\mathbf{S} = \frac{1}{N-1} \sum_{n=1}^N \tilde{\mathbf{x}}_n \tilde{\mathbf{x}}_n^\top = \frac{1}{N-1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}$$
  (Uses the $\frac{1}{N-1}$ divisor convention as in `numpy.cov` and `scikit-learn`, not $\frac{1}{N}$).
- **C4: Eigendecomposition & Deterministic Sign Rule:**
  - Use a symmetric eigensolver (e.g., `numpy.linalg.eigh`).
  - Sort eigenvalues in descending order: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$.
  - Unit-norm eigenvectors $\mathbf{u}_1, \dots, \mathbf{u}_D$ as columns of $\mathbf{U}$.
  - **Sign Flip Rule:** Flip each $\mathbf{u}_j$ so that its component with the largest absolute value is positive. This rule must be applied to every reported or compared eigenvector (including library outputs).
- **C5: Explained Variance Ratio:**
  $$r_j = \frac{\lambda_j}{\sum_{i=1}^D \lambda_i}, \quad R(k) = \sum_{j=1}^k r_j$$
- **C6: Reporting & Precision Rules:**
  - Round numbers to **4 decimal places**.
  - Residuals and differences checked against the numerical tolerance threshold ($10^{-8}$) must be reported in **scientific notation** (e.g., `3.2e-15`).
  - Two quantities "agree" when their maximum absolute difference is $< 10^{-8}$.
  - **Plot Requirements:** Every plot must include a title, labelled axes, and a legend whenever more than one series is displayed.

---

## 2. Problem Breakdown

---

### Problem 1 (P1): PCA from First Principles

#### Task Description
Derive PCA via variance maximisation and implement PCA entirely from scratch without using high-level decomposition libraries (except for validation). Form the sample covariance matrix $\mathbf{S}$, compute its eigendecomposition with sorted eigenvalues and canonical sign-flipped eigenvectors, inspect the total variance / orthonormality, and verify agreement with `sklearn.decomposition.PCA`.

#### Constraints & Specifications
- Use only the 13 feature columns of `wine_train.csv` standardised according to **C2**. Do not use `class`.
- Implementation must compute $\mathbf{S}$ (**C3**) and symmetric eigendecomposition (**C4**) from scratch.
- `sklearn.decomposition.PCA(n_components=13, svd_solver="full")` may **only** be used for verification.

#### Deliverables
1. **From-Scratch Implementation:** Python code computing $\mathbf{S}$ and eigendecomposition.
2. **Eigenvalues Output:** Print all 13 eigenvalues in descending order, one per line, with 4 decimal places in the exact format:
   ```text
   lambda 1 = x.xxxx
   lambda 2 = x.xxxx
   ...
   lambda 13 = x.xxxx
   ```
   Include this output in the report.
3. **Orthonormality & Trace Check:** Report:
   - $\|\mathbf{U}^\top \mathbf{U} - \mathbf{I}\|_F$ (Frobenius norm)
   - $\operatorname{tr}(\mathbf{S}) - \sum_{j} \lambda_j$
   - Explicit numerical value of $\operatorname{tr}(\mathbf{S})$.
4. **Library Verification:** Fit `sklearn.decomposition.PCA(n_components=13, svd_solver="full")`. Apply the sign flip rule (**C4**) to rows of `components_`. Report:
   - Maximum absolute difference between custom eigenvalues and `explained_variance_`.
   - Maximum absolute difference between custom $\mathbf{U}$ and `components_.T`.
   - Confirm that both differences are $< 10^{-8}$ (reported in scientific notation).
5. **Loadings Table & Top Features:**
   - Table of loadings for $\mathbf{u}_1$ and $\mathbf{u}_2$ indexed by feature name.
   - Names of the top 3 features with largest $|\text{loading}|$ on PC1 and PC2.

---

### Problem 2 (P2): Variance Captured vs. Number of Components

#### Task Description
Analyze the variance retained and reconstruction error as the number of retained principal components $k \in \{1, \dots, 13\}$ varies. Evaluate component retention heuristics (Kaiser criterion, elbow criterion, cumulative variance thresholds) and prove/verify the algebraic identity relating reconstruction error to discarded eigenvalues.

#### Constraints & Specifications
- Standardised training data (**C2**); test data transformed using training statistics ($\boldsymbol{\mu}, \mathbf{s}$) and training eigenvectors $\mathbf{U}$ (**C1**).
- Projection and reconstruction for standardised $\mathbf{z}_n$:
  $$\mathbf{c}_n = \mathbf{U}_k^\top \mathbf{z}_n, \quad \hat{\mathbf{z}}_n = \mathbf{U}_k \mathbf{c}_n = \mathbf{U}_k \mathbf{U}_k^\top \mathbf{z}_n$$
- Reconstruction error on set $\mathcal{D}$ with $n_{\mathcal{D}}$ points:
  $$J_{\mathcal{D}}(k) = \frac{1}{n_{\mathcal{D}} - 1} \sum_{n \in \mathcal{D}} \|\mathbf{z}_n - \hat{\mathbf{z}}_n\|^2$$
  ($n_{\text{train}} = 124$, $n_{\text{test}} = 54$).
- **Retention Criteria Definitions:**
  - Cumulative variance thresholds: $k_\tau = \min \{k : R(k) \ge \tau\}$ for $\tau \in \{0.80, 0.90, 0.95\}$.
  - Kaiser criterion: Retain components where $\lambda_j > \frac{\operatorname{tr}(\mathbf{S})}{D}$ (which simplifies to $\lambda_j > 1$ for standardized data).
  - Elbow criterion: $k_{\text{elbow}} = \max \{k : r_k \ge 0.10\}$ (last component explaining $\ge 10\%$ of total variance).

#### Deliverables
1. **Plots:**
   - **Scree plot:** Bar chart of $r_j$ vs. $j$ with a vertical dashed line at $k_{\text{elbow}}$.
   - **Cumulative variance plot:** Curve of $R(k)$ vs. $k$ with horizontal dashed reference lines at $0.80$, $0.90$, and $0.95$.
   - **Reconstruction error plot:** $J_{\text{train}}(k)$ and $J_{\text{test}}(k)$ plotted against $k$ on the same axes.
2. **Criteria Values:**
   - Stated value of $k_{\text{elbow}}$.
   - Reported values of $k_{0.80}$, $k_{0.90}$, $k_{0.95}$, and number of components retained by the Kaiser criterion.
3. **Summary Table:** A table for all $k \in \{1, \dots, 13\}$ with columns:
   - $r_k$
   - $R(k)$
   - $J_{\text{train}}(k)$
   - $J_{\text{test}}(k)$
4. **Reconstruction Identity Verification:**
   - Compute:
     $$\max_{k=1,\dots,13} \left| J_{\text{train}}(k) - \sum_{j=k+1}^D \lambda_j \right|$$
   - Report in scientific notation and verify it is $< 10^{-8}$.
5. **Short Discussion Paragraph (max 5 sentences):**
   - (i) Compare $k_{\text{elbow}}$ with Kaiser count and $k_{0.80}, k_{0.90}, k_{0.95}$, commenting on what $R(k_{\text{elbow}})$ indicates about this choice.
   - (ii) Discuss whether $J_{\text{test}}(k)$ is non-increasing in $k$, justifying algebraically from $\|\mathbf{z}_n - \hat{\mathbf{z}}_n\|^2 = \sum_{j>k} (\mathbf{u}_j^\top \mathbf{z}_n)^2$ and noting what the plot shows.

---

### Problem 3 (P3): Effect of Preprocessing, and the Two-Dimensional Projection

#### Task Description
Investigate the sensitivity of PCA to data scaling by evaluating three distinct preprocessing pipelines. Project the data onto the first two principal components, inspect class separability visually and via a downstream 5-Nearest Neighbour (5-NN) classifier, and explain why high variance does not necessarily equate to discriminative class structure.

#### Constraints & Specifications
- Compare three preprocessing variants (fitted on training data only per **C1**):
  - **V1 (Centred only):** Mean subtraction $\mathbf{x}_n - \bar{\mathbf{x}}$, no scaling.
  - **V2 (Min–Max):** Scale to $[0, 1]$ using training min/max: $x'_{nj} = \frac{x_{nj} - \min_j}{\max_j - \min_j}$, followed by mean-centring the scaled features.
  - **V3 (Z-score):** Standardisation as in **C2**.
- Apply custom PCA code from P1 (**C3**, **C4**) to each preprocessed variant.
- Preprocess test points using training-set parameters of the respective variant.
- 2D projection: $\mathbf{c}_n = \mathbf{U}_2^\top \mathbf{x}'_n \in \mathbb{R}^2$.
- Classifier: `KNeighborsClassifier(n_neighbors=5)` with Euclidean distance and default settings, trained on 2D training projections and scored on 2D test projections.

#### Deliverables
1. **Comparison Table:** One row per variant (`V1`, `V2`, `V3`) with columns:
   - $r_1$
   - $r_2$
   - $r_1 + r_2$
   - Feature with largest $|\text{loading}|$ on PC1
   - 5-NN test classification accuracy on the 2D projection.
2. **V1 Feature Variance Report:** Report the 3 features with the largest raw sample variance.
3. **Three-Panel Figure:** One figure with three panels (V1, V2, V3) showing:
   - PC1 vs. PC2 projections.
   - Training points as filled circles, test points as crosses ($\times$).
   - Points coloured by ground-truth class label $\{0, 1, 2\}$.
4. **Short Discussion Paragraph (max 6 sentences):**
   - (i) Why does V1 produce an $r_1$ value so drastically different from V3?
   - (ii) V1 captures significantly more variance in 2 components than V3; is it better for classification? What does this imply about equating variance captured with useful task structure?
   - (iii) Which preprocessing variant should be chosen before PCA on this data, and why?

---

### Problem 4 (P4): PCA versus Feature Selection for a Downstream Classifier

#### Task Description
Compare dimensionality reduction via unsupervised feature extraction (PCA) against supervised feature selection techniques: a filter method (ANOVA $F$-score) and a wrapper method (Sequential Forward Selection, SFS). Evaluate performance using multinomial logistic regression across subset sizes $k \in \{1, 2, 3, 5, 8\}$.

#### Constraints & Specifications
- Standardised features (**C2**), training statistics only (**C1**).
- **Classifier:** `LogisticRegression(C=1.0, max_iter=1000)` from `sklearn.linear_model` (multinomial, default L2, default solver).
- **Cross-Validation Folds:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` on the training set, instantiated once and reused for all evaluations.
- **Methods for $k \in \{1, 2, 3, 5, 8\}$:**
  - **(A) PCA:** First $k$ principal component scores $\mathbf{U}_k^\top \mathbf{z}_n$ with $\mathbf{U}_k$ from P1.
  - **(B) Filter (F-score):** Select top $k$ features ranked in descending order by ANOVA $F$-score:
    $$F_j = \frac{\sum_{c=1}^C N_c (m_{jc} - m_j)^2 / (C - 1)}{\sum_{c=1}^C \sum_{n \in c} (z_{nj} - m_{jc})^2 / (N - C)}$$
    where $C = 3$, $N_c$ is class sample count, $m_{jc}$ is class mean, $m_j$ is overall mean.
  - **(C) Wrapper (Sequential Forward Selection):** Greedy forward selection starting from $\mathcal{F} = \emptyset$. At each step, add the candidate $j \notin \mathcal{F}$ maximizing mean 5-fold CV accuracy on the training set. Break ties by smallest column index. Stop at $k$ features. (Equivalent to `sklearn.feature_selection.SequentialFeatureSelector` with `direction="forward", scoring="accuracy", cv=folds`).
- **Baseline:** Logistic regression trained on all 13 standardised features without reduction, evaluated on the test set.

#### Deliverables
1. **Accuracy Table:** For each $k \in \{1, 2, 3, 5, 8\}$, report test accuracy across three columns:
   - Method (A): PCA
   - Method (B): Filter ($F$-score)
   - Method (C): Wrapper (SFS)
2. **Plot & Baseline Metric:**
   - Report baseline test accuracy (scalar).
   - Plot of Test Accuracy vs. $k$ showing three curves for (A), (B), and (C), with the baseline drawn as a dashed horizontal line.
3. **Feature Rankings & Overlap Analysis:**
   - Full ranking of all 13 features by $F$-score (names, highest score first).
   - Names of features selected by (B) and (C) for $k = 3$ and $k = 8$, along with the count of shared features.
4. **Short Discussion Paragraph (max 6 sentences):**
   - (i) Smallest $k$ in the table where each method achieves accuracy within one test sample of baseline ($\ge \text{baseline} - 1/54$).
   - (ii) PCA is unsupervised (never sees labels) whereas Filter and Wrapper are supervised; explain how the plot reflects (or does not reflect) this distinction.
   - (iii) Given the test sample size of $N_{\text{test}} = 54$, discuss confidence in ranking the methods based on the observed margins.

