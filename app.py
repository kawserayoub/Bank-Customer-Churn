import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import logging

# Load the trained model
model = load_model('churn.keras')
print("Model loaded successfully!")

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Function to preprocess user input
def preprocess_input(data):
    data = pd.DataFrame(data, index=[0])
    data = pd.get_dummies(data, columns=['country', 'gender'], drop_first=True)

    # Assuming the same columns used in training
    scaler = MinMaxScaler()
    cols_to_scale = ['credit_score', 'balance', 'estimated_salary']
    data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

    return data

st.title('Bank Customer Churn Prediction')

# Create input fields
credit_score = st.number_input('Credit Score')
country = st.selectbox('Country', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
balance = st.number_input('Balance')
estimated_salary = st.number_input('Estimated Salary')
age = st.number_input('Age')
tenure = st.number_input('Tenure')
num_of_products = st.number_input('Number of Products')
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Preprocess input and make prediction
input_data = {
    'credit_score': credit_score,
    'country': country,
    'gender': gender,
    'balance': balance,
    'estimated_salary': estimated_salary,
    'age': age,
    'tenure': tenure,
    'num_of_products': num_of_products,
    'has_cr_card': has_cr_card,
    'is_active_member': is_active_member
}

if st.button('Predict'):
    try:
        input_df = preprocess_input(input_data)
        prediction = model.predict(input_df)
        prediction = (prediction > 0.5).astype(int)

        if prediction == 1:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')

        logging.info(f"Prediction made with input: {input_data} - Result: {prediction}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"Error making prediction: {e}")
