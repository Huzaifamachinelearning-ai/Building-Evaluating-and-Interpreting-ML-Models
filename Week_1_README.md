# Project 1 — ML-Powered Customer Analytics
## Week 1: Exploratory Data Analysis (EDA)

**Telco Customer Churn Dataset**  
*Introduction to Applied AI*

**Student:** Muhammad Huzaifa  
**Instructor:** Dr. Faiz Ahmed

---

## Project Overview

This project explores the **Telco Customer Churn** dataset to understand which customer characteristics are associated with leaving a telecommunications service. Week 1 focuses on loading and inspecting the data, checking data quality, summarizing churn, and communicating patterns through visualizations.

> **Scope note:** This is exploratory analysis. Observed relationships are not proof of causation, and the findings should be validated before being used for business decisions or predictive modeling.

## Objectives

- Load and inspect the dataset using **Pandas** and **NumPy**.
- Review column types, dataset structure, summary statistics, and missing or invalid values.
- Measure the distribution of churned and retained customers.
- Visualize relationships between churn and tenure, charges, contract type, internet service, and payment method.
- Examine correlations among numeric features.
- Identify questions to investigate during the next stage of the project.

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Analysis workflow |
| Pandas | Data loading, cleaning, grouping, and cross-tabulation |
| NumPy | Numeric operations |
| Matplotlib | Plot creation and layout |
| Seaborn | Statistical visualizations |

## Dataset

- **Name:** Telco Customer Churn
- **File used:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target column:** `Churn`
- **Important fields explored:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, and `PaymentMethod`

The notebook loads the CSV from a Kaggle input path. If running locally, update the `pd.read_csv(...)` path to the location of your downloaded CSV file.

## Workflow

1. **Import libraries** — load the analysis and visualization packages.
2. **Load and inspect data** — read the CSV, view the first rows, and inspect the schema.
3. **Check data quality** — count missing values and convert `TotalCharges` to numeric values; invalid entries are coerced to missing values.
4. **Summarize churn** — calculate the number and percentage of customers who churned or stayed.
5. **Explore numeric variables** — inspect tenure and monthly/total charges against churn status.
6. **Explore customer categories** — compare churn across contract types, internet service categories, and payment methods.
7. **Review correlations** — map selected numeric feature correlations.
8. **Summarize findings** — record initial patterns and questions for follow-up analysis.

## Visual Results

The figures below are extracted from the Week 1 notebook outputs.

### 1. Customer Tenure

![Customer tenure distribution and tenure by churn status]![alt text]<img width="941" height="382" alt="image" src="https://github.com/user-attachments/assets/afe1dee4-f694-434b-af10-cf2c27b6ac7a" />


**What to look for:** Compare tenure distributions for customers who stayed and customers who churned. Shorter tenure appears to be associated with greater churn risk in the notebook's interpretation.

### 2. Monthly Charges

![Monthly charges distribution and monthly charges by churn status]![alt text]<img width="846" height="480" alt="image-1" src="https://github.com/user-attachments/assets/127eb3c3-9b3e-45df-b096-523f5e79371a" />


**What to look for:** Review the distribution of monthly charges and compare the two churn groups. Higher monthly charges are identified in the notebook as a possible churn-associated characteristic, not as a proven cause.

### 3. Total Charges

![Total charges by churn and tenure versus total charges]![alt text]<img width="991" height="402" alt="image-2" src="https://github.com/user-attachments/assets/3077f500-239d-422e-ac3e-5a6e9886b709" />


**What to look for:** Total charges tend to accumulate with tenure. The analysis excludes rows where `TotalCharges` could not be converted to a valid numeric value.

### 4. Contract Type

![Contract type distribution and churn by contract type]<img width="846" height="446" alt="image-3" src="https://github.com/user-attachments/assets/a79d300e-1fed-48e9-8525-14c8211edae8" />
<img width="820" height="478" alt="image-4" src="https://github.com/user-attachments/assets/678a4c78-6807-4815-9ca6-ada1328056ae" />


**What to look for:** The notebook reports higher churn among month-to-month customers than among customers with longer contracts.

### 5. Internet Service

![Churn by internet service type]![alt text]<img width="1033" height="397" alt="image-5" src="https://github.com/user-attachments/assets/cd7df195-e107-41b4-b6aa-de1afbc9187f" />


**What to look for:** The notebook identifies fiber-optic customers as a group with comparatively high churn. Further analysis is needed to test what may explain this relationship.

### 6. Payment Method

![Churn rate by payment method]![alt text]<img width="862" height="480" alt="image-6" src="https://github.com/user-attachments/assets/9a515444-75d2-4f7e-afc7-30ed7ae1a3ad" />


**What to look for:** Electronic-check users show a relatively high churn rate in the notebook's analysis.

### 7. Feature Correlations

![Correlation heatmap of numeric features]![alt text]<img width="600" height="520" alt="image-7" src="https://github.com/user-attachments/assets/e9b532e8-e842-4951-a599-fa3d6f74ed16" />


**What to look for:** Correlation helps reveal relationships among numeric variables. It does not establish causation, and categorical columns require suitable encoding before they can be included in this analysis.

## Key Findings Recorded in the Notebook

The notebook's EDA summary highlights these initial patterns:

- **Contract:** Month-to-month customers show the highest churn risk; longer contracts are associated with better retention.
- **Tenure:** Customers with shorter tenure appear more likely to churn.
- **Charges:** Higher monthly charges are associated with increased churn in the analysis.
- **Internet service:** Fiber-optic customers show comparatively high churn.
- **Payment method:** Electronic-check users show comparatively high churn.
- **Additional services:** Online Security and Tech Support are suggested as features worth investigating in relation to retention.
- **Overall churn:** The notebook reports **1,869 churned customers out of 7,043**, approximately **26.5%**.

These are descriptive findings from the notebook and should be confirmed with refreshed calculations, statistical testing, and—later—model evaluation.

## Data Preparation Notes

- `TotalCharges` is converted with `pd.to_numeric(..., errors="coerce")`.
- Rows with invalid/missing `TotalCharges` are excluded from the total-charge plots.
- Selected binary columns (`Churn`, `Partner`, `Dependents`, `PhoneService`, and `PaperlessBilling`) are mapped from `Yes`/`No` to `1`/`0` for the correlation heatmap.
- Before modeling, verify missing values, duplicates, data types, class balance, and any preprocessing that could cause data leakage.

## Week 1 Deliverables

- Dataset loaded and inspected.
- Initial data-quality checks completed.
- Churn distribution summarized.
- Visual analysis completed for tenure, charges, contract type, internet service, and payment method.
- Correlation heatmap generated.
- Initial findings and Week 2 questions documented.

## Conclusion

Week 1 establishes an exploratory baseline for the customer-churn project. The notebook identifies contract type, tenure, monthly charges, internet service, and payment method as useful areas for deeper investigation. The next step is to validate these patterns and prepare a reproducible machine-learning workflow with an appropriate train/test split and evaluation metrics.

---

<div align="center">

**Project 1 · Week 1 · Exploratory Data Analysis**

</div>
