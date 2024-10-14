import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging

# Load the trained model
model = load_model('streamlit/churn.keras')

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='w' 
)

# Function to preprocess user input
def preprocess_input(data):
    data = pd.DataFrame(data, index=[0])
    data = pd.get_dummies(data, columns=['country', 'gender'], drop_first=True)

    expected_columns = [
        'credit_score', 'balance', 'estimated_salary', 'age', 'tenure', 'num_of_products',
        'has_cr_card', 'is_active_member', 'country_Germany', 'country_Spain', 'gender_Male'
    ]

    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_columns]

    scaler = MinMaxScaler()
    cols_to_scale = ['credit_score', 'balance', 'estimated_salary']
    data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

    return data

# Header for the app
st.title('Bank Customer Churn Prediction')

# Create a form to enter customer details
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input('Credit Score (300-850)', min_value=300, max_value=850, value=300, help="Range: 300 (Poor) to 850 (Excellent)")
        country = st.selectbox('Country', ['Select', 'France', 'Germany', 'Spain'], help="Select the customer's country.")
        gender = st.selectbox('Gender', ['Select', 'Male', 'Female'], help="Select the customer's gender.")
        balance = st.number_input('Balance (€)', min_value=0.0, value=0.0, help="Enter the customer's account balance (€).")

    with col2:
        estimated_salary = st.number_input('Estimated Salary (€)', min_value=0.0, value=0.0, help="Enter the customer's estimated annual salary (€).")
        age = st.number_input('Age', min_value=18, max_value=100, value=18, help="Enter the customer's age.")
        tenure = st.number_input('Years with Bank', min_value=0, max_value=10, value=0, help="How many years has the customer been with the bank?")
        num_of_products = st.number_input('Products Owned', min_value=1, max_value=4, value=1, help="Includes: credit cards, savings accounts, loans, or investments.")

    # Yes/No questions with simplified labels
    has_cr_card = st.selectbox('Has Credit Card?', ['Select', 'No', 'Yes'], help="Does the customer own a credit card?")
    is_active_member = st.selectbox('Active Member?', ['Select', 'No', 'Yes'], help="Is the customer an active bank member?")

    # Submit button at the end of the form
    submit_button = st.form_submit_button(label='Check Churn Risk')

# Handle form submission
if submit_button:
    if country == 'Select' or gender == 'Select' or has_cr_card == 'Select' or is_active_member == 'Select':
        st.error("Please complete all fields.")
    else:
        # Map 'Yes'/'No' to 1/0
        input_data = {
            'credit_score': credit_score,
            'country': country,
            'gender': gender,
            'balance': balance,
            'estimated_salary': estimated_salary,
            'age': age,
            'tenure': tenure,
            'num_of_products': num_of_products,
            'has_cr_card': 1 if has_cr_card == 'Yes' else 0,
            'is_active_member': 1 if is_active_member == 'Yes' else 0
        }

        # Preprocess the input data
        input_df = preprocess_input(input_data)

        # Make prediction using the model
        prediction = model.predict(input_df)
        prediction = (prediction > 0.5).astype(int)

        # Output the prediction result with box and emojis
        with st.container():
            if prediction == 1:
                st.write('🚨 **Customer at risk of churn.**')
                st.warning('Recommended: Review for retention strategies.')
            else:
                st.write('✅ **Customer retention likely.**')
                st.success('No immediate action required.')

