# Credit Default Risk Scoring System
## Overview
Banks and lending companies lose millions every year by approving customers who 
default on their loans. At the same time, overly strict rules reject good customers 
who would have repaid — losing potential revenue.

This project tackles that problem using machine learning. I built a credit default 
prediction model that scores every applicant on their likelihood of defaulting, then 
used a cost-based analysis to find the exact approval threshold that saves the most 
money — not just the threshold that looks best on paper.

## Business Problem
Most lending decisions are still made using flat rules — if income is above X, approve. 
These rules miss the bigger picture. A customer can have decent income but a history 
of late payments that signals trouble ahead.

The goal was simple: can we do better than flat rules, and can we prove it in dollars?

## What I Did
- Cleaned and explored 150,000 customer records using Python and SQL
- Engineered 6 new features from raw data including late payment trends and utilization buckets
- Built and compared 3 models: Logistic Regression, Random Forest and XGBoost
- Used a cost matrix to find the optimal approval threshold
- Validated model stability using 5-fold stratified cross validation
- Implemented SHAP explainability to explain individual customer risk predictions
- Quantified the business impact in dollar terms

## Results
| Metric | Value |
|---|---|
| Best Model | XGBoost |
| AUC Score | 0.868 |
| Cross Validation AUC | (your CV mean score here) |
| Default Threshold Cost | $404,615 |
| Optimal Threshold Cost | $191,790 |
| Simulated Cost Saving | $212,825 (52% reduction) |
| Optimal Threshold | 0.06 |
| Approval Rate | 74% |

## Key Finding
Shifting the decision threshold from the standard 0.5 to 0.06 reduced total business 
cost by 52%. This works because a missed defaulter costs 16x more than a wrongly 
rejected good customer — so the model should flag even small risks.

## Tools Used
- Python — Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP
- SQL — SQLite via Jupyter Notebook
- Jupyter Notebook

## Dataset
Give Me Some Credit — Kaggle
150,000 customers, 11 features, binary default label
Link: https://www.kaggle.com/c/GiveMeSomeCredit

## How to Run
1. Download dataset from Kaggle link above
2. Place cs-training.csv in the project folder
3. Open Credit_Default_Risk_Scoring_System.ipynb  in Jupyter Notebook
4. Run all cells top to bottom

## Project Structure
- Credit_Default_Risk_Scoring_System.ipynb — main notebook (cleaning, EDA, features, models, SHAP)
- SQlite.ipynb — SQL exploration notebook
- model_results.csv — predictions and scores
- Business Case Statement.docx — business case and problem statement
- 07_shap_summary.png — SHAP explainability chart

## AXP CFR Alignment
- "Economic logic to enable profitable decisions" → cost matrix threshold optimizer saving $212,825
- "Analyze large amounts of data to derive business insights" → SQL EDA on 150,000 customer records
- "Development and validation of predictive models" → 3 models built, compared and cross-validated
- "Derive business insights and create innovative solutions" → SHAP explainability + feature importance
- Charts — all visualization PNGs
- README.md — this file
