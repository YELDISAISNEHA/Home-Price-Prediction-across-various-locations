import streamlit as st
import json
import pandas as pd
import pickle

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('banglore_data.csv')  # Ensure this file contains location, total_sqft, bath, price, bhk

df = load_data()

# Streamlit app
st.title("House Price Prediction")

# Location selection
location = st.selectbox("Select Location", df['location'].unique())

# Get minimum values for the selected location
filtered_df = df[df['location'] == location]
min_total_sqft = int(filtered_df['total_sqft'].min())
min_bhk = int(filtered_df['bhk'].min())
min_bath = int(filtered_df['bath'].min())

# User input fields with default values set to minimums
total_sqft = st.number_input("Total Square Feet", min_value=min_total_sqft, value=min_total_sqft)
bhk = st.number_input("BHK", min_value=min_bhk, value=min_bhk, step=1)
bath = st.number_input("Bathrooms", min_value=min_bath, value=min_bath, step=1)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[total_sqft, bhk, bath]], columns=['total_sqft', 'bhk', 'bath'])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Predicted Price: ₹{predicted_price:,.2f} lakhs")
