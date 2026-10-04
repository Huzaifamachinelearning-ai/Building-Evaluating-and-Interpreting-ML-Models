# ML-Powered Customer Analytics: Telco Customer Churn

End-to-end machine learning project that predicts customer churn for a telecom provider. It covers exploratory analysis, model building and evaluation, hyperparameter optimization, customer segmentation, and a serialized production-ready model.

| | |
|---|---|
| **Author** | Muhammad Huzaifa |
| **Course** | Introduction to Applied AI (BS 7th Semester) |
| **Instructor** | Dr. Faiz Ahmed |
| **Dataset** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, 7,043 customers, 21 columns) |
| **Environment** | Kaggle Notebooks, Python 3 |
| **Final model** | Tuned XGBoost, test ROC-AUC **0.848** |

---

## Week 1: Exploratory Data Analysis 

**Notebook:** `week-1-exploratory-data-analysis-on-kaggle.ipynb`  

### Objectives
Load, clean, and visualize the data to find patterns between customer features and churn.

### What was done
- Inspected structure, dtypes, and summary statistics.
- Fixed `TotalCharges`: it is stored as text and contains blank values, so it was converted with `pd.to_numeric(..., errors='coerce')`.
- Measured the class balance of the target.
- Visualized numeric features (tenure, monthly charges, total charges) with histograms, boxplots, violin plots, and scatter plots.
- Compared churn rates across contract type, internet service, and payment method.
- Built a correlation heatmap after encoding binary columns as 0/1.

### Findings: high-risk customer profile

| Factor | Observation |
|---|---|
| Contract | Month-to-month customers churn the most |
| Tenure | New customers are far more likely to leave |
| Monthly charges | Higher monthly charges go with higher churn |
| Internet service | Fiber optic users churn more than other groups |
| Payment method | Electronic check users have a relatively high churn rate |
| Add-on services | Customers without Online Security or Tech Support are higher risk |

`tenure` and `TotalCharges` are strongly related, since longer-staying customers accumulate more charges.

---

## Week 2: Building, Evaluating and Interpreting Models

**Notebook:** `week-2-building-evaluating-and-interpreting-ml.ipynb`

### Pipeline

1. **Preprocessing:** Imputed the 11 blank `TotalCharges` values (all with `tenure = 0`) with the median, dropped `customerID`, and one-hot encoded categoricals (`drop_first=True`).
2. **Split:** Stratified 80/20 split (5,634 train / 1,409 test, `random_state=42`).
3. **Baseline:** A "most frequent" dummy classifier reaches **73.5% accuracy** while catching **0 churners**. Every real model has to beat this.
4. **Logistic Regression** inside a `Pipeline` with `StandardScaler`.
5. **Interpretability:** Odds-ratio table, plus a manual sigmoid check that reproduced scikit-learn's probability exactly (z = -3.070, p = 0.0444).
6. **Confusion matrix and metrics computed by hand** (TN = 925, FP = 110, FN = 162, TP = 212).
7. **ROC-AUC and threshold analysis**, then a **cost-based threshold**.
8. **Decision Trees and Random Forest**, including an overfitting study, out-of-bag score, and permutation importance.
9. **Class imbalance and feature engineering:** `class_weight='balanced'`, plus engineered features `n_services`, `is_new`, `charge_per_mo`, `price_jump`.

### Interpretation (Logistic Regression odds ratios)

| Protective factors | Odds ratio | Risk factors | Odds ratio |
|---|---|---|---|
| `tenure` | 0.295 | `InternetService_Fiber optic` | 2.179 |
| `MonthlyCharges` | 0.398 | `TotalCharges` | 1.644 |
| `Contract_Two year` | 0.555 | `StreamingMovies_Yes` | 1.295 |

> Fiber optic service more than doubles the odds of churn, holding the other features constant.

### Business-cost threshold

A missed churner costs **PKR 6,000** and an unnecessary retention offer costs **PKR 1,000**.

- Theoretical optimal threshold: `C_FP / (C_FP + C_FN)` = **0.14**
- Empirical optimum on the cost curve: **0.15**, much lower than the default 0.50.

| Threshold | Flagged | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.2 | 685 | 0.467 | 0.856 | 0.604 |
| 0.3 | 544 | 0.518 | 0.754 | 0.614 |
| 0.4 | 441 | 0.567 | 0.668 | 0.613 |
| 0.5 | 322 | 0.658 | 0.567 | 0.609 |
| 0.6 | 210 | 0.710 | 0.398 | 0.510 |
| 0.7 | 89 | 0.730 | 0.174 | 0.281 |

### Overfitting study (Decision Tree)
Test accuracy peaks around depth 6 (0.797). With no depth limit, train accuracy is 0.998 against 0.742 test, a gap of about 25.6 points.

### Model comparison (hold-out test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| LR (balanced) | 0.739 | 0.505 | 0.781 | 0.613 | 0.841 |
| Decision Tree (depth 5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| Random Forest | 0.807 | 0.673 | 0.529 | 0.593 | 0.842 |

### Week 2 conclusions
- Accuracy hides the failure on churners: the default Logistic Regression misses 162 of 374 churners.
- `class_weight='balanced'` raised recall from 0.567 to 0.781 and cut precision from 0.658 to 0.505.
- Engineered features did not help Random Forest (AUC 0.8422 to 0.8420), because trees already learn interactions on their own.
- Permutation importance ranks `tenure`, `TotalCharges`, and `Contract_Two year` highest. It disagrees with impurity-based importance (MDI) on features like fiber optic, since MDI is biased toward high-cardinality or heavily split features.
- **Recommendation:** Logistic Regression with a **0.15 threshold**. AUC ties with Random Forest, and it is explainable to business stakeholders.

---

## Week 3: Model Optimization and Unsupervised Learning

**Notebook:** `week-3-model-optimization-and-unsupervised-learni.ipynb`

The test set is **locked away until the final step** (Part 7), and model selection is done purely with cross-validation.

### Part 1: Split noise
Training the same model on 20 different splits gave accuracy from **0.780 to 0.828** (std 0.0104). This matches the theoretical standard error (0.0107). A single split is not a reliable basis for comparing models.

### Part 2: 5-fold stratified cross-validation

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 | 0.594 |
| Random Forest | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 |

### Part 3: Hyperparameter tuning
- **Validation curve (Logistic Regression `C`):** CV AUC plateaus near 0.846 for `C` between 0.1 and 10, with a small train/CV gap (0.850 vs 0.846).
- **Grid search (Random Forest):** 24 combinations, 120 fits, about 96 s. Best CV AUC **0.8468** (`max_depth=8`, `min_samples_leaf=20`, `max_features='sqrt'`).
- **Random search (Random Forest):** 24 iterations, 120 fits, about 115 s. Best CV AUC **0.8464**.
- Random search is preferable when tuning many hyperparameters. Four values for each of six hyperparameters would need 4,096 combinations and 20,480 fits with a grid.

### Part 4: XGBoost
- Early stopping chose **247 trees** (up to 2,000 allowed) with validation AUC **0.8541**.
- Train and validation log-loss curves were plotted to show where overfitting begins.
- Randomized search (30 iterations) over 7 hyperparameters gave best CV AUC **0.8502**, with a shallow model (`max_depth=2`, learning rate about 0.034).
- Gain-based importance highlights fiber optic internet, two-year contract, and electronic-check payment.

### Part 5: K-Means customer segmentation
Features: `tenure`, `MonthlyCharges`, `TotalCharges`, and `n_services`, all standardized. The elbow plot bends near **k = 4**. Silhouette peaks at k = 2, but two segments are too coarse for targeting. Churn was **not** used to form clusters, only to profile them.

| Segment | Customers | Avg tenure (mo) | Avg monthly ($) | Avg services | Churn rate | Suggested action |
|---|---|---|---|---|---|---|
| Cluster 1: At-risk new high-spenders | 2,157 | 18.4 | 80.41 | 3.28 | **43%** | Proactive onboarding, tailored bundles, loyalty incentives |
| Cluster 3: New budget, low engagement | 1,918 | 9.0 | 37.71 | 1.20 | **32%** | Discounted security and backup add-ons |
| Cluster 2: Loyal high-value | 1,938 | 59.8 | 92.09 | 5.06 | **14%** | VIP rewards, referral programs |
| Cluster 0: Stable long-term budget | 1,030 | 53.6 | 30.96 | 1.48 | **5%** | Automated renewals, low-touch service |

### Part 6: PCA
**15 of 30** components are needed to retain 90% of the variance, so the information is spread across many dimensions. The top PC1 loadings are dominated by the redundant "No internet service" dummy columns.

### Part 7: Final selection and a single test evaluation

| Model | CV AUC | CV Std |
|---|---|---|
| Logistic Regression (tuned C) | 0.8464 | 0.0129 |
| Random Forest (random search) | 0.8464 | 0.0114 |
| **XGBoost (tuned)** | **0.8502** | 0.0117 |

The best model by CV (XGBoost) was refit on the training data and evaluated **once** on the test set:

| Test ROC-AUC | Recall | Precision |
|---|---|---|
| **0.8483** | 0.521 | 0.659 |

The result sits inside the CV expectation of 0.8502 ± 2σ (0.827 to 0.874), so there is no sign of overfitting. The model is saved as `churn_model.joblib`.

### Concept checks covered
Standard error for model comparison, data leakage (scaler inside the `Pipeline`), grid-search fit counts, the optimism of `best_score_`, the gradient boosting mechanism, and why K-Means needs scaled features.

---

## Results Summary

| Stage | Best approach | Metric |
|---|---|---|
| Baseline | Always predict "stay" | Accuracy 0.735, recall 0 |
| Week 2 | Logistic Regression, threshold 0.15 | AUC 0.842, explainable |
| Week 3 CV | Tuned XGBoost | CV AUC 0.8502 |
| Week 3 test | Tuned XGBoost (single evaluation) | Test AUC 0.8483 |

---

## Key Takeaways

1. **Start with a baseline.** A 73.5% accuracy score is meaningless without knowing it catches no churners.
2. **Accuracy is the wrong headline metric** on imbalanced data. Use recall, precision, and AUC.
3. **Choose the threshold from business costs**, not the default 0.5.
4. **A single split is noisy.** Use stratified cross-validation, and touch the test set only once.
5. **Gains from tuning and boosting are modest** here. Simple models are already competitive, so explainability matters.
6. **Segmentation adds business value.** The highest-priority group is new customers with high spend and many services (43% churn).

---
