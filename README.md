# credit-risk-forecasting
# 💳 Credit Risk Forecasting: Predictive Default Scoring in R

![R](https://img.shields.io/badge/Language-R-blue.svg)
![Machine Learning](https://img.shields.io/badge/Task-Binary_Classification-brightgreen.svg)
![Dataset](https://img.shields.io/badge/Dataset-UCI_Credit_Card-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

An end-to-end statistical modeling and machine learning project implemented in **R** to predict customer default probability on credit cards for the subsequent month. This project addresses critical real-world banking challenges, including severe multicollinearity across billing cycles, a **78% / 22% class imbalance**, and the transformation of raw monetary exposure into relative financial stress indicators.

---

## 📌 Business & Analytical Objectives

Credit scoring systems enable financial institutions to quantify individual insolvency risks before delinquency occurs. The main objectives of this project are:
1. **Predictive Performance:** Build and benchmark classification models capable of identifying high-risk accounts.
2. **Feature Engineering:** Design domain-driven financial stress metrics to overcome the limitations of raw absolute balances.
3. **Statistical & Methodological Rigor:** Mitigate severe multicollinearity across monthly statements and handle class imbalance while preventing data leakage through a strictly train-parameterized preprocessing workflow.

---

## ⚙️ Methodology & Pipeline Workflow
* **1. Raw Data Ingestion & Cleaning**
  * Source: UCI Credit Card Dataset (~30,000 observations).
  * Removed duplicate records and recoded undocumented "ghost" categories (`EDUCATION` and `MARRIAGE`) to ensure statistical stability.
* **2. Exploratory Data Analysis (EDA)**
  * Bivariate distributions, credit limit analysis (`LIMIT_BAL`), and high-density multivariate inspection via `ggpairs`.
* **3. Feature Engineering (`MAX_UTIL`)**
  * Derived individual monthly utilization ratios and extracted peak credit strain per customer over the 6-month period.
* **4. Train/Test Partitioning**
  * Stratified 80/20 train/test split to preserve target class distribution and prevent data leakage.
* **5. Parameterized Standardization**
  * Z-score scaling parameters computed exclusively on the training set and applied downstream to the test set.
* **6. Dimensionality Reduction (PCA)**
  * Extracted `BILL_PC1` from monthly statement balances to resolve linear multicollinearity and reduce VIF.
* **7. Class Imbalance Mitigation**
  * Evaluated prior probability threshold moving (0.22), synthetic oversampling via **ROSE**, and **Youden's J Index** optimization.
* **8. Model Benchmark & Interpretability**
  * Trained and evaluated Unbalanced Logistic Regression, Balanced Logistic Regression (ROSE), Stepwise Logistic Regression (AIC selection), Average Partial Effects (APE), and Random Forest.

---
## 🔍 Key Findings & Feature Engineering

### 1. Feature Categorization & Variable Taxonomy

The dataset features are structured into four logical blocks:

- **Demographic Profile**: `SEX`, `AGE`, `MARRIAGE`, `EDUCATION` (recoded for stability).
- **Financial Capacity**: `LIMIT_BAL` (credit limit granted by the bank).
- **Behavioral History**: `PAY_0` to `PAY_6` (monthly tracking of repayment delay buckets from April to September 2005).
- **Financial Exposure**: `BILL_AMT1`–`BILL_AMT6` (monthly statements) and `PAY_AMT1`–`PAY_AMT6` (amounts paid).

### 2. Resolving Multicollinearity via PCA

Exploratory correlation analysis revealed severe linear collinearity (r ∈ [0.80, 0.95]) across statement balances `BILL_AMT1`–`BILL_AMT6`. To resolve coefficient instability in generalized linear models:

- Principal Component Analysis (PCA) was fitted strictly on the standardized training statements.
- The first principal component (`BILL_PC1`) captured the dominant variance of cumulative debt exposure.
- Redundant raw billing variables were removed, successfully suppressing Variance Inflation Factors (VIF) across all linear model coefficients.

### 3. The MAX_UTIL Indicator (Relative Financial Stress)

Absolute debt amounts (e.g., $5,000) fail to convey risk without context (a $5,000 debt on a $5,000 limit represents maximum risk, while on a $100,000 limit it represents negligible exposure).

Monthly utilization rates were computed with a defensive zero-division safeguard:
UTIL_t = BILL_AMT_t / (LIMIT_BAL + 1)

The maximum peak stress over the 6-month period was extracted using parallel maximum (`pmax()`):
MAX_UTIL = max(UTIL_1, ..., UTIL_6)

**Empirical Validation**: Defaulters exhibit a median peak usage of ~60% (vs. ~38% for non-defaulters). Furthermore, the 3rd quartile (Q3) for the default group reaches 1.0, proving that 25% of defaulting clients completely saturated or exceeded their credit limit.

## 📊 Model Evaluation & Benchmark

Models were trained and evaluated on an independent, held-out test set (N=6,000, 20% stratified partition):

| Model Architecture | Imbalance Strategy | Decision Threshold | Accuracy | Sensitivity (Recall) | Specificity | Misclass. Error | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression (Unbalanced) | Prior Probability Cutoff | 0.22 | ~0.74 | ~0.65 | ~0.76 | ~0.26 | ~0.765 |
| Stepwise Logistic Regression | AIC Parsimony + Cutoff | 0.22 | ~0.74 | ~0.65 | ~0.77 | ~0.26 | ~0.766 |
| Logistic Regression (ROSE) | Synthetic Resampling | 0.50 | ~0.75 | ~0.62 | ~0.78 | ~0.25 | ~0.765 |
| Random Forest (Default) | Standard Majority Vote | 0.50 | ~0.82 | ~0.37 | ~0.94 | ~0.18 | ~0.783 |
| Random Forest (Optimized) | Youden's J Index | 0.235 | ~0.75 | ~0.68 | ~0.77 | ~0.25 | ~0.783 |

## 💡 Core Takeaways

- **Interpretability (Logistic Regression + APE)**: Stepwise selection identified recent repayment delay (`PAY_0`), credit limit (`LIMIT_BAL`), and peak credit utilization (`MAX_UTIL`) as the primary drivers of default. Manual Average Partial Effect (APE) derivations confirmed that recent payment delinquency produces the single highest marginal increase in default probability.
- **Non-linear Capacity (Random Forest)**: Random Forest yielded the highest discriminative ability (AUC ~ 0.783). Shifting the decision threshold via Youden's J statistic resolved the tree ensemble's bias toward the majority class, raising Sensitivity from 37% to 68% with balanced specificity.
## 📁 Repository Structure

* `data/`
  * `UCI_Credit_Card.csv` — Raw dataset file (or external download link).
* `src/`
  * `credit_risk_analysis.R` — Reproducible R script containing the entire modeling pipeline.
* `presentation/`
  * `credit_risk_presentation.pdf` — Slide deck presentation covering the business and analytical insights.
* `README.md` — Project documentation and executive summary.

---
## 🛠️ Requirements & R Libraries

To reproduce this analysis locally, install the required packages:

```r
install.packages(c(
  "tidyverse",
  "dplyr",
  "caret",
  "ROSE",
  "pROC",
  "randomForest",
  "car",
  "margins",
  "corrplot",
  "GGally",
  "moments",
  "data.table"
))
```

## 👤 Author

Edoardo Boccomini - [edoardoboccomini-tech](https://github.com/edoardoboccomini-tech)
