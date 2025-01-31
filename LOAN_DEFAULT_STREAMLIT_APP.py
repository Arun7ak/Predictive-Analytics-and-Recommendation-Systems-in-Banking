#IMPORTING ALL THE LIBRARIES TO HANDLE AND LOAD THE TRAINED MODEL
import streamlit as st
import pandas as pd
import joblib 

#GIVE BACKGROUND AND STYLING FOR THE APP 
st.markdown(
    """
    <style>
    /* Background color for the entire app */
    .stApp {
        background-color: #d4edda;
    }

    /* Title styling */
    .title {
        color: #1f77b4;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }

    /* Input labels */
    label {
        color: #333333;
        font-size: 16px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Success and Error message styling */
    .success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        font-size: 18px;
        text-align: center;
    }

    .error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        font-size: 18px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#TITLE OF THE APP
st.markdown('<h1 class="title">🏦 Loan Default Prediction</h1>', unsafe_allow_html=True)

#CREATING THE USER INPUT BOX
age = st.number_input("📌 Age", min_value=18, max_value=100, step=1)
income = st.number_input("💰 Income", min_value=0.0, step=100.0)
credit_score = st.number_input("📊 Credit Score", min_value=300, max_value=850, step=1)
loan_term = st.number_input("📆 Loan Term (in months)", min_value=1, step=1)
loan_amount = st.number_input("🏦 Loan Amount", min_value=0.0, step=100.0)
monthly_installment = st.number_input("📉 Monthly Installment", min_value=0.0, step=10.0)
interest_rate = st.number_input("📈 Interest Rate (%)", min_value=0.0, step=0.1)

#DISPLAY THE ENTERED DATA BY USER
st.subheader("📌 Entered Data")
st.write({
    "Age": age,
    "Income": income,
    "Credit Score": credit_score,
    "Loan Term": loan_term,
    "Loan Amount": loan_amount,
    "Monthly Installment": monthly_installment,
    "Interest Rate": interest_rate
})

# LOAD THE TRAINED MODEL
model = joblib.load(r"C:\Users\arune\OneDrive\Desktop\TENNIS PYTHON PROJECT.pkl")

# CREATING PREDICT BUTTON AND PREDICT THE OUTPUT BASED ON TRAINED MODEL
if st.button("🔍 Predict Loan Default"):
    input_data = [[age, income, credit_score, loan_term, loan_amount, monthly_installment, interest_rate]]
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.markdown('<div class="error">🚨 The person is likely to DEFAULT on the loan.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success">✅ The person is likely to REPAY the loan.</div>', unsafe_allow_html=True)
