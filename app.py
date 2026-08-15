import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Load model
model = XGBClassifier()
model.load_model("credit_risk_model.json")

st.title('Credit Default Risk Scoring System')
st.markdown('Enter customer details below to predict default probability and get an approval decision.')

# Input fields
age = st.slider('Age', 18, 100, 35)
income = st.number_input('Monthly Income (USD)', min_value=0, value=5000)
utilization = st.slider('Credit Utilization (0 to 1)', 0.0, 1.0, 0.3)
debt_ratio = st.slider('Debt Ratio (0 to 1)', 0.0, 1.0, 0.3)
late_30 = st.number_input('Times 30-59 Days Late', min_value=0, value=0)
late_60 = st.number_input('Times 60-89 Days Late', min_value=0, value=0)
late_90 = st.number_input('Times 90+ Days Late', min_value=0, value=0)
open_loans = st.number_input('Open Credit Lines', min_value=0, value=5)
real_estate = st.number_input('Real Estate Loans', min_value=0, value=0)
dependents = st.number_input('Number of Dependents', min_value=0, value=0)

# Feature engineering
total_late = late_30 + late_60 + late_90
serious_delinquent = 1 if late_90 > 0 else 0
is_young = 1 if age < 35 else 0
debt_income_ratio = debt_ratio * income
worsening = 1 if (late_30 > 0 and late_90 > 0) else 0

# Utilization bucket dummies
util_medium = 1 if 0.3 < utilization <= 0.6 else 0
util_high = 1 if 0.6 < utilization <= 0.9 else 0
util_maxed = 1 if utilization > 0.9 else 0

# Build input dataframe
input_data = pd.DataFrame([[
    utilization, age, late_30, debt_ratio, income,
    open_loans, late_90, real_estate, late_60, dependents,
    total_late, serious_delinquent, is_young, debt_income_ratio,
    worsening, util_medium, util_high, util_maxed
]], columns=[
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents', 'total_late_payments', 'serious_delinquent',
    'is_young', 'debt_income_ratio', 'worsening_delinquency',
    'Utilization_bucket_medium', 'Utilization_bucket_high',
    'Utilization_bucket_maxed'
])

if st.button('Predict'):
    prob = model.predict_proba(input_data)[0][1]
    cost_fn = 250
    cost_fp = 15

    st.subheader(f'Default Probability: {prob:.1%}')

    if prob >= 0.06:
        st.error('Decision: REJECT — High default risk')
        st.write(f'Estimated loss if approved: USD {prob * cost_fn:.0f}')
    else:
        st.success('Decision: APPROVE — Low default risk')
        st.write(f'Estimated revenue from approval: USD {cost_fp}')