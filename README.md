# 💳 Credit Risk Prediction App  

A machine learning web application that predicts the **likelihood of loan default** using customer financial and behavioral data.  
The model helps financial institutions assess borrower risk and make informed lending decisions.  

🚀 **Live App:** [https://sreeja-credit-risk-app.streamlit.app](https://credit-risk-predictor-by-sreeja.streamlit.app/)

---

## 📘 Overview  

This project builds a **credit risk prediction model** trained on real-world loan applicant data.  
It identifies potential defaulters based on credit history, income, debt ratio, and other financial indicators.  
The final deployment uses an **XGBoost classifier** integrated with a **Streamlit web dashboard** for interactive risk analysis.  

---

## 🧠 Machine Learning Pipeline  

### 1️⃣ Data Preprocessing  
- Imputed missing values (median for numeric, mode for categorical).  
- Addressed class imbalance using **SMOTE + Tomek Links** to balance the default (minority) class.  
- Engineered key interaction features:
  - `TotalPastDue`
  - `DebtToIncome`
  - `UtilizationPerCreditLine`
  - `CreditLinesPerDependent`
  - `Age_Delinquency_Interaction`
  - `HasRealEstate`

### 2️⃣ Model Training & Comparison  
Multiple models were trained and compared — Logistic Regression, Random Forest, and XGBoost.  
XGBoost showed the best trade-off between recall and precision for credit risk classification.

| Model | Data Type | Recall | Precision | F1 | F2 | AUC |
|--------|-----------|---------|-----------|------|------|------|
| Logistic Regression | Balanced | 0.75 | 0.20 | 0.32 | 0.49 | 0.85 |
| Random Forest | Balanced | 0.30 | 0.42 | 0.35 | 0.31 | 0.83 |
| **XGBoost** | **Balanced** | **0.30** | **0.46** | **0.36** | **0.33** | **0.86** |

### 3️⃣ Model Evaluation  
- Optimized the decision threshold for **maximum F2 score** (to prioritize recall).  
- Best model (XGBoost):
  - **Recall:** 0.81  
  - **AUC:** 0.856  
  - **Optimized F2 Score:** 0.511  
- Used **SHAP values** for explainability and feature importance analysis.

---

## 🌐 Streamlit Web App  

The deployed Streamlit app allows users to input borrower information and receive:
- **Predicted Default Probability**  
- **Credit Score Estimate (300–900)**  
- **Risk Category:** 🟢 Low | 🟠 Medium | 🔴 High  

### 🎛 Dashboard Features
- Clean and responsive UI with sidebar inputs  
- Instant model inference using the deployed XGBoost model  
- Auto-calculated credit score for easy interpretation  
- Real-time feedback with detailed borrower summary  

---

## 🏗️ Tech Stack  

| Category | Tools |
|-----------|--------|
| Programming | Python |
| Libraries | Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, SHAP |
| Frontend | Streamlit |
| Version Control | Git & GitHub |
| Deployment | Streamlit Cloud |

---

## 🧩 Repository Structure  

📦 credit-risk-app
├── app.py # Streamlit web application
├── Credit-risk.ipynb # Jupyter Notebook (model development)
├── xgboost_final_credit.pkl # Trained XGBoost model
├── README.md # Project documentation
└── images/ # Logo and visuals


---

## 📊 Key Results  

- **Best Model:** XGBoost (Balanced dataset)  
- **AUC:** 0.856  
- **Recall:** 0.81  
- **F2 Score:** 0.511  
- **KS Statistic:** 0.558  
- **Feature Importance Highlights:**  
  - Age and delinquency frequency  
  - Revolving credit utilization  
  - Debt-to-income ratio  
  - Real estate ownership  

---

## 👩‍💻 Author  

**Sreeja Gantyada**  
M.Tech – Computer Science and Engineering  
IIT (ISM) Dhanbad  

📧 [LinkedIn Profile](https://www.linkedin.com/in/sreejagantyada)  
📍 Developed as part of independent research and applied ML practice in **Credit Risk Modeling**.

---

## 🧾 License  

This repository is open for educational and academic use.  
If you find it helpful, please ⭐ **star the repo** or cite it in your work.  

---


