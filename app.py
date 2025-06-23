import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Term Deposit Subscription Predictor", layout="centered")
st.title("Term Deposit Subscription Predictor")
st.write("""
This app predicts whether a client will subscribe to a term deposit based on their information and campaign details.
""")

# Load model
def load_model():
    return joblib.load('simple_term_deposit_model.pkl')

model = load_model()

# Feature options (from bank-names.txt and data)
job_options = [
    "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur",
    "student", "blue-collar", "self-employed", "retired", "technician", "services"
]
marital_options = ["married", "divorced", "single"]
education_options = ["unknown", "secondary", "primary", "tertiary"]
default_options = ["yes", "no"]
housing_options = ["yes", "no"]
loan_options = ["yes", "no"]
contact_options = ["unknown", "telephone", "cellular"]
month_options = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
poutcome_options = ["unknown", "other", "failure", "success"]

# Input form
with st.form("input_form"):
    st.subheader("Client Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", job_options)
    marital = st.selectbox("Marital Status", marital_options)
    education = st.selectbox("Education", education_options)
    default = st.selectbox("Has credit in default?", default_options)
    balance = st.number_input("Average yearly balance (euros)", value=1000)
    housing = st.selectbox("Has housing loan?", housing_options)
    loan = st.selectbox("Has personal loan?", loan_options)

    st.subheader("Last Contact Information")
    contact = st.selectbox("Contact communication type", contact_options)
    day = st.number_input("Last contact day of the month", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last contact month", month_options)
    duration = st.number_input("Last contact duration (seconds)", min_value=0, value=100)

    st.subheader("Campaign Information")
    campaign = st.number_input("Number of contacts during this campaign", min_value=1, value=1)
    pdays = st.number_input("Days since last contact (-1 if not previously contacted)", value=-1)
    previous = st.number_input("Number of contacts before this campaign", min_value=0, value=0)
    poutcome = st.selectbox("Outcome of previous campaign", poutcome_options)

    submitted = st.form_submit_button("Predict Subscription")

if submitted:
    # Prepare input as DataFrame
    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    input_df = pd.DataFrame([input_dict])
    # Predict
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]
    st.markdown("---")
    st.subheader("Prediction Result")
    if pred == 1:
        st.success(f"The client is likely to SUBSCRIBE to a term deposit. (Probability: {pred_proba:.1%})")
    else:
        st.error(f"The client is NOT likely to subscribe. (Probability: {pred_proba:.1%})")
    st.markdown("---")
    st.info("**Business Insights:**\n\n- Students and retirees have highest success rates.\n- Tertiary education clients are more likely to subscribe.\n- Longer calls generally lead to better outcomes.\n- Multiple campaigns reduce success rates.\n- Focus on quality over quantity in calls.") 