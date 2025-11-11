# -*- coding: utf-8 -*-
"""
Credit Risk Prediction App (XGBoost Model)
Created by: Sreeja, IIT ISM Dhanbad
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------
# 🧭 Page Setup
# -----------------------------------------------------
st.set_page_config(page_title="Credit Risk Estimator", page_icon="💰", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2b4b80;'>💳 Credit Risk Estimator</h1>", unsafe_allow_html=True)
st.caption("An interactive dashboard to assess credit default probability using a trained XGBoost model.")

# -----------------------------------------------------
# ⚙️ Load Model
# -----------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_final_credit.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -----------------------------------------------------
# 🧾 Sidebar
# -----------------------------------------------------
with st.sidebar:
    st.image("images/logo.png", caption="Lauki Finance", width=160)
    st.markdown("### 📘 Instructions")
    st.write("""
    1. Enter or adjust details below.  
    2. Click **Evaluate Risk** to view credit prediction.  
    3. Results include default probability, credit score, and risk rating.
    """)
    st.markdown("---")
    st.info("Powered by Machine Learning (XGBoost)")

# -----------------------------------------------------
# 👤 Section 1: Borrower Profile
# -----------------------------------------------------
st.subheader("👤 Borrower Information")

col1, col2, col3 = st.columns(3)
age = col1.number_input("Age (Years)", min_value=18, max_value=100, value=32)
MonthlyIncome = col2.number_input("Monthly Income ($)", min_value=0.0, max_value=100000.0, value=4500.0, step=500.0)
NumberOfDependents = col3.number_input("Dependents", min_value=0, max_value=10, value=1, step=1)

# -----------------------------------------------------
# 💳 Section 2: Credit Profile
# -----------------------------------------------------
st.subheader("🏦 Credit History")

col4, col5, col6 = st.columns(3)
RevolvingUtilizationOfUnsecuredLines = col4.slider("Revolving Utilization (%)", 0.0, 10.0, 0.45)
DebtRatio = col5.number_input("Debt Ratio", min_value=0.0, max_value=10.0, value=0.8)
NumberOfOpenCreditLinesAndLoans = col6.number_input("Open Credit Lines & Loans", min_value=0, max_value=50, value=5)

col7, col8, col9 = st.columns(3)
NumberOfTimes90DaysLate = col7.number_input("Times 90+ Days Late", min_value=0, max_value=100, value=0)
NumberOfTime30_59DaysPastDueNotWorse = col8.number_input("Times 30–59 Days Past Due", min_value=0, max_value=100, value=0)
NumberOfTime60_89DaysPastDueNotWorse = col9.number_input("Times 60–89 Days Past Due", min_value=0, max_value=100, value=0)

col10, col11 = st.columns(2)
NumberRealEstateLoansOrLines = col10.number_input("Real Estate Loans/Lines", min_value=0, max_value=20, value=1)
loan_purpose = col11.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Auto", "Other"])

# -----------------------------------------------------
# 🧮 Derived Features
# -----------------------------------------------------
TotalPastDue = (
    NumberOfTime30_59DaysPastDueNotWorse +
    NumberOfTimes90DaysLate +
    NumberOfTime60_89DaysPastDueNotWorse
)
DebtToIncome = DebtRatio / (MonthlyIncome + 1e-6)
UtilizationPerCreditLine = RevolvingUtilizationOfUnsecuredLines / (NumberOfOpenCreditLinesAndLoans + 1)
CreditLinesPerDependent = (NumberOfOpenCreditLinesAndLoans + 1) / (NumberOfDependents + 1)
HasRealEstate = 1 if NumberRealEstateLoansOrLines > 0 else 0
Age_Delinquency_Interaction = age * (NumberOfTimes90DaysLate + 1)

input_data = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": NumberOfTime30_59DaysPastDueNotWorse,
    "DebtRatio": DebtRatio,
    "MonthlyIncome": MonthlyIncome,
    "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
    "NumberOfTimes90DaysLate": NumberOfTimes90DaysLate,
    "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
    "NumberOfTime60-89DaysPastDueNotWorse": NumberOfTime60_89DaysPastDueNotWorse,
    "NumberOfDependents": NumberOfDependents,
    "TotalPastDue": TotalPastDue,
    "DebtToIncome": DebtToIncome,
    "UtilizationPerCreditLine": UtilizationPerCreditLine,
    "CreditLinesPerDependent": CreditLinesPerDependent,
    "HasRealEstate": HasRealEstate,
    "Age_Delinquency_Interaction": Age_Delinquency_Interaction
}])

# -----------------------------------------------------
# 🚀 Predict Button
# -----------------------------------------------------
st.markdown("---")
if st.button("💡 Evaluate Risk"):
    try:
        prob = model.predict_proba(input_data)[:, 1][0]
        credit_score = int(900 - (prob * 600))

        if prob < 0.3:
            rating = "Excellent"
            risk = "🟢 Low Risk"
            msg = "Strong borrower profile — safe to approve."
        elif prob < 0.6:
            rating = "Moderate"
            risk = "🟠 Medium Risk"
            msg = "Caution advised — verify credit reports or collateral."
        else:
            rating = "Poor"
            risk = "🔴 High Risk"
            msg = "High probability of default — loan not recommended."

        st.markdown("### 🧮 Credit Evaluation Summary")
        st.success("✅ Risk Evaluation Completed Successfully!")

        colA, colB, colC = st.columns(3)
        colA.metric("Default Probability", f"{prob:.2%}")
        colB.metric("Credit Score Estimate", credit_score)
        colC.metric("Rating", rating)
        st.markdown(f"### {risk}")
        st.info(msg)

        with st.expander("📋 View Detailed Input Data"):
            st.dataframe(input_data.style.format("{:.2f}"))

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")

# -----------------------------------------------------
# 🧠 Footer
# -----------------------------------------------------
st.markdown("---")
st.caption("© 2025 | Credit Risk Modeling with XGBoost | Developed by Sreeja, IIT ISM Dhanbad")
