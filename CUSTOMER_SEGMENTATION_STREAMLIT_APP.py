#IMPORTING ALL THE LIBRARIES TO HANDLE AND LOAD THE TRAINED MODEL
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

#GIVE BACKGROUND AND STYLING FOR THE APP 
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f1f5f9; /* Light Grey Background */
    }
    .title {
        color: #4e73df;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-top: 50px;
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
    .stTextInput>div>input {
        font-size: 16px;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .stTextInput>div>input:focus {
        border-color: #4e73df;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#TITLE OF THE APP
st.markdown('<h1 class="title">💳 Loan Default Prediction</h1>', unsafe_allow_html=True)

#CREATING THE USER INPUT BOX
st.markdown('<div class="input-container">', unsafe_allow_html=True)
total_transactions = st.number_input("Total Transactions", min_value=0, step=1)
total_amount = st.number_input("Total Amount", min_value=0.0, step=100.0)
num_deposits = st.number_input("Number of Deposits", min_value=0, step=1)
num_withdrawals = st.number_input("Number of Withdrawals", min_value=0, step=1)

#DISPLAY THE ENTERED DATA BY USER
st.write("Entered Data:")
st.write({
    "Total Transactions": total_transactions,
    "Total Amount": total_amount,
    "Number of Deposits": num_deposits,
    "Number of Withdrawals": num_withdrawals
})

st.markdown('</div>', unsafe_allow_html=True)


#LOAD THE MODEL IF IT EXIST AND PREDICT THE OUTPUT, IF NOT EXIST GIVE ERROR
try:
    model1 = joblib.load(r"D:\BANK PROJECT.pkl")  
    scaler = joblib.load(r"d:\BANK PROJECT\kmeans.pkl") 

    if not hasattr(model1, "predict"):
        raise ValueError("The loaded object is not a valid model. Please check the file.")

    #CREATING PREDICT BUTTON
    if st.button("🔍 Predict Cluster"):
        input_data = np.array([[total_transactions, total_amount, num_deposits, num_withdrawals]])
        input_data = scaler.transform(input_data)  
        
        input_data_list = input_data.tolist()
        cluster = model1.predict(input_data)[0]
        st.markdown(f'<div class="stSuccess">✅ The person belongs to **Cluster {cluster}**</div>', unsafe_allow_html=True)

except Exception as e:
    st.markdown(f'<div class="stError">❌ Error: {str(e)}</div>', unsafe_allow_html=True)





