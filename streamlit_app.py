import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline,CustomData
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


st.title("Fraud Detection Prediction")
name=st.text_input('Enter your name')
email=st.text_input('Enter your email address')

if name:
    st.write(f'hello {name} welcome to the app')
# Header with an explanation of the app
st.header("Input Parameters")
st.write("Please provide the transaction details below. Each field is explained for your convenience.")

# Select box for 'type'
type_options = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
type_ = st.selectbox(
    "Select Transaction Type:", 
    type_options,
    help="Choose the type of transaction. For example, 'CASH-IN' means the user is depositing money into the account."
)
st.write("""
    - **PAYMENT**: A transaction made to pay for services or goods.
    - **TRANSFER**: A transfer of funds between two accounts.
    - **CASH_OUT**: A withdrawal of funds from the account.
    - **DEBIT**: A direct payment made from the account.
    - **CASH_IN**: A deposit of funds into the account.
""")



# Integer inputs for other parameters with descriptions
step = st.number_input(
    "Enter Step:", 
    min_value=1, max_value=1000, value=1, step=1,
    help="Step refers to the time step in the transaction log, it helps track the sequence of the transaction."
)
st.write("Step represents the time in a sequence of transactions. Higher values indicate later transactions.")

amount = st.number_input(
    "Enter Amount:", 
    min_value=1, max_value=1000000, value=100, step=10,
    help="The monetary value of the transaction. A higher amount might indicate suspicious activity."
)
st.write("The transaction amount (in dollars or local currency). Larger amounts may trigger higher suspicion.")

nameOrigFreq = st.number_input(
    "Enter Frequency of 'nameOrig':", 
    min_value=0, max_value=1000, value=10, step=1,
    help="The frequency of the 'nameOrig' account. This value indicates how often this account has been involved in transactions."
)
st.write("Frequency of the origin account ('nameOrig') in past transactions. A high frequency may indicate frequent activities.")

oldbalanceOrg = st.number_input(
    "Enter Old Balance of 'nameOrig':", 
    min_value=0, max_value=1000000, value=50000, step=100,
    help="The account balance before the transaction occurred. This can help assess if the balance is consistent with the transaction size."
)
st.write("The previous balance of the origin account ('nameOrig'). Large transactions relative to balance may be suspicious.")

nameDestFreq = st.number_input(
    "Enter Frequency of 'nameDest':", 
    min_value=0, max_value=1000, value=5, step=1,
    help="The frequency of the 'nameDest' account. This value indicates how often this account has been involved in transactions."
)
st.write("Frequency of the destination account ('nameDest') in past transactions. A sudden increase in frequency may indicate fraud.")

oldbalanceDest = st.number_input(
    "Enter Old Balance of 'nameDest':", 
    min_value=0, max_value=1000000, value=50000, step=100,
    help="The account balance of the 'nameDest' account before the transaction. This can help assess if the balance is consistent with the transaction size."
)
st.write("The previous balance of the destination account ('nameDest'). Discrepancies between balance and transaction amount may raise red flags.")
data=CustomData(step, type_, amount, nameOrigFreq, oldbalanceOrg, nameDestFreq, oldbalanceDest)
input_dataframe=data.get_data_as_frame()
# Predict Button
if st.button('Predict'):
    # Call the dummy prediction function (Replace with your actual model)
    
    st.write('The datas you have input are:')
    input_dataframe

    prediction=PredictPipeline()
    preds=prediction.predict(input_dataframe)
    if preds ==0:
        st.success('It is a Non-Fraudulent Transaction')
    else:
        st.error('It is a Fraudulent Transaction')

  