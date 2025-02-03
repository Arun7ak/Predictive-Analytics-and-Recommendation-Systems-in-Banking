import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

#USING CSS FOR STYLE THE APP
st.markdown(
    """
    <style>
    .stApp {
        background-color:#90EE90; 
    }
    .title {
        color: #4e73df;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-top: 50px;
    }
    .stNumberInput>label {
    font-size: 20px;  
    font-weight: bold;
    color: black;  
    }

    .input-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        margin-bottom: 40px;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #2e5bff;
    }
    .stNumberInput>div>input,
    .stTextInput>div>input {
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .stNumberInput>div>input:focus,
    .stTextInput>div>input:focus {
        border-color: #4e73df;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724; 
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        padding: 15px;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# TITLE OF THE APP
st.markdown('<h1 class="title">📊 CUSTOMER SEGMENTATION BASED ON TRANSACTIONS</h1>', unsafe_allow_html=True)


# CREATING THE USER INPUT BOX
st.markdown('<div class="input-container">', unsafe_allow_html=True)

total_no_of_transactions = st.number_input("Total Number of Transactions", min_value=0, step=1)
total_amount_in_dollar = st.number_input("Total Amount in Dollar", min_value=0.0, step=100.0)
num_of_deposits = st.number_input("Number of Deposits", min_value=0, step=1)
num_of_withdrawals = st.number_input("Number of Withdrawals", min_value=0, step=1)
withdrawals_amount_in_dollar = st.number_input("Withdrawals Amount in Dollar", min_value=0.0, step=100.0)
deposits_amount_in_dollar = st.number_input("Deposits Amount in Dollar", min_value=0.0, step=100.0)

# DISPLAY THE ENTERED DATA BY USER
st.write("Entered Data:")
st.write({
    "Total Number of Transactions": total_no_of_transactions,
    "Total Amount in Dollar": total_amount_in_dollar,
    "Number of Deposits": num_of_deposits,
    "Number of Withdrawals": num_of_withdrawals,
    "Withdrawals Amount in Dollar": withdrawals_amount_in_dollar,
    "Deposits Amount in Dollar": deposits_amount_in_dollar
})

st.markdown('</div>', unsafe_allow_html=True)

# LOAD THE MODEL IF IT EXISTS AND PREDICT THE OUTPUT
scaler = joblib.load(r"d:\BANK PROJECT\scaled_d.pkl")
model1 = joblib.load(r"d:\BANK PROJECT\kmeans.pkl")

# CREATING PREDICT BUTTON
if st.button("🔍 Predict Cluster"):
    input_data = np.array([[total_no_of_transactions, total_amount_in_dollar, num_of_deposits,
                                num_of_withdrawals, withdrawals_amount_in_dollar, deposits_amount_in_dollar]])

    input_data_scaled = scaler.transform(input_data)

    cluster = model1.predict(input_data_scaled)[0]
    st.markdown(f'<div class="stSuccess">✅ The person belongs to **Cluster {cluster}**</div>', unsafe_allow_html=True)


# LOADING THE DATASET FOR PREVIEW
df = pd.read_csv(r"d:/BANK PROJECT/cleaned_loan_dataset.csv")
st.subheader("📊 Customer Segmentation Preview")
st.write(df)  












