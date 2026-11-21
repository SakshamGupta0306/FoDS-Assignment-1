# Assignment 2: Principal Component Analysis (FoDS)

This repository contains the complete implementation, scripts, figures, and comprehensive report for **FoDS Assignment 2 (Principal Component Analysis)**.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+

### Setting Up the Virtual Environment
To create and activate the virtual environment, run:

```bash
# 1. Navigate to the Assignment2 directory
cd "FoDS ASS 2/FoDS-Assignment-1/Assignment2"

# 2. Create the virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install required packages
pip install -r requirements.txt
```

---

## 2. Project Structure

```text
Assignment2/
├── wine_train.csv             # Fixed training dataset (N=124, D=13)
├── wine_test.csv              # Fixed test dataset (N_test=54, D=13)
├── common.py                  # Core utilities & C1-C6 mandatory conventions
├── p1_pca_scratch.py          # Problem 1: PCA from first principles
├── p2_variance_components.py  # Problem 2: Scree, retention criteria, reconstruction error
├── p3_preprocessing_study.py  # Problem 3: Preprocessing (V1, V2, V3) and 2D projections
├── p4_feature_selection.py    # Problem 4: PCA vs. Filter (F-score) vs. Wrapper (SFS)
├── requirements.txt           # Virtual environment dependencies
├── ASSIGNMENT_SPEC.md         # Detailed task constraints and deliverables
├── REPORT.md                  # Complete answers, tables, and discussions for your submission
└── output/                    # Generated figures and charts:
    ├── p2_scree_cumulative.png
    ├── p2_reconstruction_error.png
    ├── p3_projections_comparison.png
    └── p4_accuracy_vs_k.png
```

---

## 3. Running the Code Files

Make sure your virtual environment is active (`source .venv/bin/activate`), then run each task:

### Run Problem 1 (PCA from Scratch & Verification)
```bash
python p1_pca_scratch.py
```
*Outputs: Printed eigenvalues, orthonormality Frobenius norm, trace identity check, scikit-learn verification, and PC1/PC2 loadings table.*

### Run Problem 2 (Variance Captured vs. Components)
```bash
python p2_variance_components.py
```
*Outputs: Summary table ($k=1\dots13$), elbow criterion, Kaiser count, cumulative variance thresholds, algebraic reconstruction identity verification ($< 10^{-8}$), and plots in `output/`.*

### Run Problem 3 (Preprocessing Study & 2D Projection)
```bash
python p3_preprocessing_study.py
```
*Outputs: Comparison table for V1, V2, V3, top raw-variance features, 5-NN test accuracies, and 3-panel projection plot in `output/`.*

### Run Problem 4 (PCA vs. Feature Selection)
```bash
python p4_feature_selection.py
```
*Outputs: Baseline logistic regression accuracy, full $F$-score feature ranking, test accuracies across $k \in \{1, 2, 3, 5, 8\}$, feature overlap analysis, and accuracy vs. $k$ plot in `output/`.*

---

## 4. What Else is Required for Submission?

Apart from the code files, your submission typically requires:
1. **Written Report / PDF:**
   - Include all printed eigenvalues in descending order (`lambda 1 = ...`).
   - Include all verification numbers, norms, and differences in scientific notation.
   - Include all tables (Loadings table, P2 summary table, P3 comparison table, P4 accuracy table and feature rankings).
   - Include the generated plots located in `output/`:
     - Scree plot & Cumulative variance plot (`p2_scree_cumulative.png`)
     - Reconstruction error plot (`p2_reconstruction_error.png`)
     - Three-panel 2D projections (`p3_projections_comparison.png`)
     - Test accuracy vs. $k$ curve with baseline (`p4_accuracy_vs_k.png`)
   - Include the 3 required discussion paragraphs (for P2, P3, and P4).
   - *(Everything you need for the report is ready in [REPORT.md](REPORT.md)).*
2. **Commit / Archive:** Check your course portal / LMS instructions (whether to submit a single PDF report or a `.zip` containing code + report + figures).

