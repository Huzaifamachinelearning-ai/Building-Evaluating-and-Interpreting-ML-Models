# Week 3: Model Optimization and Unsupervised Learning

**Telco Customer Churn: Cross-Validation, Hyperparameter Tuning, XGBoost, K-Means and PCA**

| | |
|---|---|
| **Student** | Muhammad Huzaifa |
| **Course** | Introduction to Applied AI |
| **Instructor** | Dr. Faiz Ahmed |
| **Dataset** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 20 raw features) |
| **Notebook** | `week-3-model-optimization-and-unsupervised-learning.ipynb` |

---

## Data Preparation

- `TotalCharges` converted to numeric; blanks (customers with `tenure = 0`) filled with `0`.
- `customerID` dropped; categorical features one-hot encoded (`drop_first=True`) giving **30 features**.
- Target: `Churn == "Yes"` mapped to `1`.
- **80/20 stratified split** (`random_state=42`): **5,634 train / 1,409 test**. The test set is not touched until Part 7.

---

## Part 1: Split Noise

The same Logistic Regression model was trained on **20 different random splits** of the training data.

<img width="452" height="347" alt="image-8" src="https://github.com/user-attachments/assets/7134084b-95bf-467a-883b-7d6a4651c36a" />


| Metric | Value |
|---|---|
| Accuracy range | 0.780 to 0.828 (a 4.8-point swing from the seed alone) |
| Empirical std | 0.0104 |
| Theoretical standard error | 0.0107 |
| 95% confidence interval | ±0.021 |

**Insight:** The empirical spread matches the theoretical standard error, so the variation is ordinary sampling noise. A single train/validation split is not a trustworthy basis for comparing models, which motivates cross-validation.

## Part 2: Cross-Validation Baseline

5-fold stratified CV on the training set (`shuffle=True`, `random_state=42`). Scaling lives inside a `Pipeline` to prevent leakage.

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.846 ± 0.013 | 0.545 ± 0.042 | 0.594 ± 0.030 |
| Random Forest (300 trees) | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 ± 0.020 |

Both models rank customers almost identically (overlapping AUC intervals). Logistic Regression catches more churners; Random Forest is more stable across folds.

---

## Part 3: Hyperparameter Tuning

### 3.1 Validation curve for Logistic Regression `C`

<img width="457" height="353" alt="image" src="https://github.com/user-attachments/assets/43cc5e3e-f6ab-48dd-85da-f8b9dde933b3" />


- **Underfitting** at strong regularization (`C < 1e-2`): both train and CV AUC fall to about 0.83.
- **Plateau** for `0.1 <= C <= 10`: CV AUC about 0.846 vs train about 0.850.
- **Best C = 10.0**. The small train/CV gap shows very little overfitting.

### 3.2 Grid vs. random search for Random Forest

| Method | Search space | Fits | Time | Best CV AUC | Best parameters |
|---|---|---|---|---|---|
| `GridSearchCV` | 24 combinations | 120 | 113 s | **0.8468** | `max_depth=8`, `min_samples_leaf=20`, `max_features='sqrt'` |
| `RandomizedSearchCV` | 24 samples | 120 | 115 s | 0.8464 | `max_depth=15`, `min_samples_leaf=15`, `max_features≈0.213` |

**Insight:** Both reach almost the same score, and the top random-search configurations all sit at 0.845 to 0.846, so the model is robust to the exact settings. With six hyperparameters at four values each, a grid would need 4⁶ = 4,096 combinations (20,480 fits). Random or Bayesian search is the practical choice at that scale.

---

## Part 4: XGBoost

### 4.1 Early stopping

- Class imbalance handled with `scale_pos_weight = 2.77`.
- Up to 2,000 trees allowed; early stopping selected **247 trees**.
- Validation ROC-AUC: **0.8541**.

<img width="477" height="350" alt="image" src="https://github.com/user-attachments/assets/06d311a8-2603-4765-8e51-a20e5269f970" />


Training loss keeps falling while validation loss flattens near the early-stopping point, the classic sign that more trees would start fitting noise.

### 4.2 Randomized search (30 iterations, 5-fold CV)

- **Tuned CV AUC: 0.8502**
- Best configuration: `max_depth=2`, `learning_rate≈0.034`, `n_estimators=476`, `subsample=0.6`, `colsample_bytree≈0.561`, `min_child_weight=1`, `reg_lambda≈1.974`.
- A shallow, slowly-learning ensemble generalized best.

### 4.3 Feature importance (gain)

<img width="625" height="336" alt="image" src="https://github.com/user-attachments/assets/e9eacb30-bc11-49c0-ac86-3101a658ff09" />


The strongest drivers of churn are **fiber-optic internet**, **contract type** and **electronic-check payment**. Several "No internet service" indicators also rank highly, but they are largely redundant with one another (see PCA below).

---

## Part 5: K-Means Customer Segmentation

**Features:** `tenure`, `MonthlyCharges`, `TotalCharges`, and `n_services` (count of subscribed add-on services), standardized with `StandardScaler` because K-Means relies on Euclidean distance. Churn was **not** used to form clusters, only to profile them afterwards.

### Choosing k

<img width="747" height="302" alt="image" src="https://github.com/user-attachments/assets/1c1dca21-ec4b-4ec6-876a-1907c0dd6236" />


The elbow bends around **k = 4**. Silhouette peaks at k = 2, but two segments are too coarse for targeted action, so **k = 4** is chosen as the balance between cluster quality and business usefulness.

### Segment profiles

| Segment | Customers | Avg. tenure (mo) | Monthly ($) | Avg. services | Churn rate | Suggested action |
|---|---:|---:|---:|---:|---:|---|
| **Cluster 1: At-risk new high-spenders** | 2,157 | 18.4 | 80.41 | 3.28 | **43%** | Proactive onboarding, bundle offers, early-lifecycle loyalty incentives |
| **Cluster 3: New budget / low engagement** | 1,918 | 9.0 | 37.71 | 1.20 | **32%** | Introductory discounts on security and backup services |
| **Cluster 2: Loyal high-value VIPs** | 1,938 | 59.8 | 92.09 | 5.06 | **14%** | VIP rewards and referral programs |
| **Cluster 0: Stable long-term budget** | 1,030 | 53.6 | 30.96 | 1.48 | **5%** | Automated renewals and self-service, minimal spend |

**Priority segment:** Cluster 1 pays a lot, uses several services and still churns at 43%, making it the best target for retention spend.

> The notebook also includes a hand-worked K-Means check on the points {1, 2, 3, 10, 11, 12} with k = 2, converging to WCSS = 4.0 in three iterations. It shows how much a poor initialization can inflate early WCSS (89.2), which is why `n_init=10` is used.

---

## Part 6: Principal Component Analysis

### 6.1 Scree plot

<img width="490" height="342" alt="image" src="https://github.com/user-attachments/assets/b45b6534-7a26-4203-83d0-6574f0d098e1" />


**15 of 30** components are needed to retain 90% of the variance, so information is spread across many dimensions rather than a few dominant ones.

### 6.2 Customers in two dimensions

<img width="537" height="360" alt="image" src="https://github.com/user-attachments/assets/60a212e8-c5fe-4af7-b591-a5c0132da5d6" />


Churners (red) and non-churners (blue) overlap heavily in the first two components, so two dimensions are not enough to separate the classes. The top PC1 loadings (about 0.302 each) are all "No internet service" indicators (`InternetService_No`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, `DeviceProtection` and `OnlineBackup`), which move together and are highly redundant.

---

## Part 7: Final Selection and Test Evaluation

### Cross-validated comparison (selection by CV only)

| Model | CV AUC | CV std |
|---|---:|---:|
| Logistic Regression (tuned C) | 0.8464 | 0.0129 |
| Random Forest (random search) | 0.8464 | 0.0114 |
| **XGBoost (tuned)** | **0.8502** | 0.0117 |

### Test set, evaluated exactly once

| Metric | Score |
|---|---:|
| **ROC-AUC** | **0.8483** |
| Recall | 0.521 |
| Precision | 0.659 |

The expected CV range is 0.8502 ± 2(0.0117) = **[0.8268, 0.8736]**. The test AUC of 0.8483 sits comfortably inside it, so there is no evidence of overfitting or test-set leakage.

The final model is serialized to **`churn_model.joblib`**.

---

## Key Takeaways

1. **One split is not enough.** Seed choice alone moved accuracy by almost 5 points, so use cross-validation.
2. **Prevent leakage.** Keep scalers inside the `Pipeline` so each fold fits them only on its own training data.
3. **Tuning has diminishing returns here.** Logistic Regression, Random Forest and XGBoost land within about 0.004 AUC of each other. XGBoost's edge is real but small.
4. **Early stopping and shallow trees beat brute force.** 247 trees (or 476 at depth 2) generalize better than very deep or very long boosting.
5. **Selection bias is real.** `best_score_` is optimistic because it is the maximum over many tried configurations, which is why the test set is used only once, at the end.
6. **Segments are actionable.** New, high-spending customers (Cluster 1) are the highest churn risk at 43%.

