# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------------
# Load trained models
# ------------------------------
rf_bank = joblib.load("bank_model.pkl")
rf_sg = joblib.load("sg_model.pkl")

# ------------------------------
# Load merged dataset for EDA
# ------------------------------
merged_data = pd.read_csv("merged_data_cleaned.csv")

# ------------------------------
# App Title
# ------------------------------
st.title("Bank & SGData Prediction Dashboard")
st.write("Interactive dashboard: Predict Bank Deposits & SG outcomes, see EDA visuals and metrics.")

# ------------------------------
# Sidebar: Bank Inputs
# ------------------------------
st.sidebar.header("Bank Data Input")
age = st.sidebar.number_input("Age", 18, 100, 30, key="age_bank")
balance = st.sidebar.number_input("Balance", 0, 1000000, 1000, key="balance_bank")
duration = st.sidebar.number_input("Last Call Duration", 0, 5000, 100, key="duration_bank")
job = st.sidebar.selectbox("Job (Bank)", merged_data['job'].unique(), key="job_bank")
marital = st.sidebar.selectbox("Marital Status (Bank)", merged_data['marital'].unique(), key="marital_bank")
education_bank = st.sidebar.selectbox("Education (Bank)", merged_data['education'].unique(), key="edu_bank")
default = st.sidebar.selectbox("Default?", merged_data['default'].unique(), key="default_bank")
housing = st.sidebar.selectbox("Housing Loan?", merged_data['housing'].unique(), key="housing_bank")
loan = st.sidebar.selectbox("Personal Loan?", merged_data['loan'].unique(), key="loan_bank")

bank_input = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "duration": [duration],
    "job": [job],
    "marital": [marital],
    "education": [education_bank],
    "default": [default],
    "housing": [housing],
    "loan": [loan]
})

# ------------------------------
# Sidebar: SGData Inputs
# ------------------------------
st.sidebar.header("SGData Input")
sex = st.sidebar.selectbox("Sex", merged_data['sex'].unique(), key="sex_sg")
marital_sg = st.sidebar.selectbox("Marital Status (SG)", merged_data['marital_status'].unique(), key="marital_sg")
education_sg = st.sidebar.selectbox("Education (SG)", merged_data['education'].unique(), key="edu_sg")
income = st.sidebar.number_input("Income", 0, 1000000, 5000, key="income_sg")
occupation = st.sidebar.selectbox("Occupation", merged_data['occupation'].unique(), key="occupation_sg")
settlement_size = st.sidebar.selectbox("Settlement Size", merged_data['settlement_size'].unique(), key="settlement_sg")

sg_input = pd.DataFrame({
    "sex": [sex],
    "marital_status": [marital_sg],
    "education": [education_sg],
    "income": [income],
    "occupation": [occupation],
    "settlement_size": [settlement_size]
})

# ------------------------------
# Make Predictions
# ------------------------------
st.header("Predictions")

if st.button("Predict Bank Deposit", key="btn_bank"):
    bank_pred = rf_bank.predict(bank_input)
    st.success(f"Bank Deposit Prediction: {bank_pred[0]}")

if st.button("Predict SGData Outcome", key="btn_sg"):
    sg_pred = rf_sg.predict(sg_input)
    st.success(f"SGData Predicted Value: {sg_pred[0]:.2f}")

# ------------------------------
# EDA Visuals
# ------------------------------
st.header("EDA Visuals & Metrics")

# Histogram of Bank Balance
st.subheader("Bank Balance Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(merged_data['balance'], kde=True, ax=ax1)
st.pyplot(fig1)

# Bank Deposit Counts
st.subheader("Bank Deposit Counts")
fig2, ax2 = plt.subplots()
sns.countplot(x='deposit', data=merged_data, ax=ax2)
st.pyplot(fig2)

# SGData Income Distribution
st.subheader("SGData Income Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(merged_data['income'], kde=True, ax=ax3)
st.pyplot(fig3)

# Settlement Size Counts
st.subheader("Settlement Size Counts")
fig4, ax4 = plt.subplots()
sns.countplot(x='settlement_size', data=merged_data, ax=ax4)
st.pyplot(fig4)

# Dataset Summary Metrics
st.subheader("Dataset Summary Metrics")
st.write(merged_data.describe())
