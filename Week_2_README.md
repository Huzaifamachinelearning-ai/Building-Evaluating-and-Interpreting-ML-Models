# Building-Evaluating-and-Interpreting-ML-Models
# 📉 Customer Churn Prediction — Building, Evaluating & Interpreting ML Models

A complete, business-grounded machine learning workflow for predicting customer churn on the **Telco Customer Churn** dataset — from a naive baseline through Logistic Regression, Decision Trees, and Random Forests, with cost-sensitive threshold tuning and a final model recommendation.

> Built as part of an *Introduction to AI* course project (Week 2: Building, Evaluating, and Interpreting ML Models), this notebook is structured the way a real churn-prevention pipeline should be: **baseline → model → interpret → evaluate → optimize for business cost → compare → decide.**

---

## 📌 Project Overview

Customer churn is one of the highest-leverage problems in subscription and telecom businesses — retaining an existing customer is almost always cheaper than acquiring a new one. This project builds a model that flags customers likely to churn **before they leave**, so a business can proactively intervene with retention offers.

The workflow answers four questions an ML engineer is always asked:

1. **Is the model actually better than doing nothing?** (baseline comparison)
2. **Can we explain *why* it makes a prediction?** (odds ratios, sigmoid math, feature importance)
3. **What's the real-world cost of being wrong?** (false negatives vs. false positives)
4. **Which model should actually go into production?** (accuracy vs. explainability trade-off)

---

## 🗂️ Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset, via Kaggle)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (Yes / No)
- **Features:** Demographics, account info (tenure, contract type, payment method), and subscribed services (phone, internet, streaming, security add-ons)
- **Cleaning notes:** `TotalCharges` is stored as text with 11 blank rows (all customers with `tenure = 0`); these are coerced to numeric and imputed with the median.

---

## 🧠 Methodology

| Part | What happens |
|---|---|
| **1. Preprocessing & Baseline** | Clean data, one-hot encode categoricals, stratified 80/20 train-test split, and a `DummyClassifier` baseline ("always predict stay") |
| **2. Logistic Regression** | Trained in a `StandardScaler → LogisticRegression` pipeline; odds ratios extracted per feature; sigmoid output verified by hand against `sklearn` |
| **3. Confusion Matrix & Metrics by Hand** | Precision, recall, and F1 recomputed manually from the confusion matrix to show *why raw accuracy is misleading* on imbalanced data |
| **4. ROC-AUC & Business-Cost Thresholding** | ROC curve and AUC; precision/recall swept across thresholds; optimal decision threshold derived from real retention-offer costs, not the default 0.5 |
| **5. Decision Trees & Random Forests** | Overfitting demonstrated across tree depths; a regularized tree visualized; a Random Forest trained with out-of-bag scoring; MDI vs. permutation feature importance compared |
| **6. Class Imbalance & Feature Engineering** | `class_weight='balanced'` trade-off (precision ↓, recall ↑); engineered features (`n_services`, `is_new`, `charge_per_mo`, `price_jump`) tested against raw features |
| **7. Compare, Document, Share** | Final side-by-side model comparison table and a deployment recommendation |

---

## 📊 Key Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline (always "stay") | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | **0.842** |
| Logistic Regression (balanced) | 0.739 | 0.505 | 0.781 | 0.613 | 0.841 |
| Decision Tree (depth = 5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| **Random Forest** | 0.807 | 0.673 | 0.529 | 0.593 | **0.842** |

Logistic Regression and Random Forest tie for the best AUC (0.842), with Random Forest edging ahead on precision and Logistic Regression (balanced) trading precision for much higher recall.

### Confusion Matrix — Logistic Regression (default threshold)
<img width="667" height="458" alt="image" src="https://github.com/user-attachments/assets/cbe0536d-e1b0-4bd2-ba29-ccba789c45ca" />



- **925** correctly predicted to stay · **212** correctly caught churners
- **162** churners missed (false negatives) · **110** false alarms (false positives)
- Accuracy alone hides the fact that the model misses ~43% of actual churners — this is why precision/recall matter more than accuracy on imbalanced churn data.

### ROC Curve
<img width="641" height="490" alt="image" src="https://github.com/user-attachments/assets/75af7c6c-93db-481a-9f66-590e2d2572d5" />


An AUC of **0.842** means the model ranks a random churner above a random non-churner about 84% of the time — regardless of what decision threshold is chosen.

### Cost-Optimized Decision Threshold
<img width="692" height="452" alt="image" src="https://github.com/user-attachments/assets/cb6430d8-f0a3-4a58-ab75-8faf8ab0a6b2" />


Retention decisions aren't free, and the two error types don't cost the same:

- **Missed churner (False Negative):** PKR 6,000 (lost customer)
- **Unneeded retention offer (False Positive):** PKR 1,000 (wasted discount)

Solving for the cost-minimizing threshold gives **t\* ≈ 0.14–0.15** (vs. the default 0.5) — moving the model to be far more sensitive to at-risk customers, because missing a churner is 6× more expensive than a false alarm.

### Decision Tree Overfitting
<img width="675" height="483" alt="image" src="https://github.com/user-attachments/assets/4e4ea4f1-a5b7-49ee-9c96-098250d77ca5" />


Training accuracy climbs toward ~99.8% as trees grow unconstrained, while test accuracy peaks around depth 6 (≈0.797) and then degrades — a textbook bias-variance trade-off, and the justification for regularizing with `max_depth` and `min_samples_leaf`.

### Regularized Decision Tree (depth = 3, for interpretability)
<img width="1377" height="517" alt="image" src="https://github.com/user-attachments/assets/0f3268c0-ce5a-43dd-9a86-999db0276a03" />


---

## 💡 Business Interpretation

- **Strongest protective factor:** longer **tenure** and **two-year contracts** — loyal, locked-in customers rarely churn.
- **Strongest risk factor:** **Fiber Optic internet service** more than doubles the odds of churning (odds ratio ≈ 2.18) — likely tied to pricing or service satisfaction issues worth investigating separately.
- **Class imbalance trade-off:** balancing class weights trades precision for recall (0.658 → 0.505 precision, 0.567 → 0.781 recall) — useful if the business would rather over-flag than miss customers.
- **Feature engineering verdict:** hand-engineered features (services count, tenure buckets, price jump) did **not** improve the Random Forest's AUC (0.8422 → 0.8420) — tree ensembles already capture these interactions natively from raw features.

---

## 🏆 Final Recommendation

**Model:** Logistic Regression
**Threshold:** 0.15 (cost-optimized, not the default 0.5)

| Criterion | Reasoning |
|---|---|
| **Predictive power** | Ties Random Forest on AUC (0.842) — no accuracy sacrificed |
| **Explainability** | Coefficients convert directly to odds ratios managers can act on (e.g., *"Fiber optic customers are ~2.2× more likely to churn"*) — Random Forest stays a black box |
| **Business alignment** | Threshold tuned to real retention-offer economics, not an arbitrary 0.5 cutoff |

---
