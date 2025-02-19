import streamlit as st
import json
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as file:
    model = pickle.load(file)
with open("columns.json", "r") as file:
    data_columns = json.load(file)

data_columns = data_columns['data_columns']
location_columns = data_columns[3:]

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
if st.button("Predict"):
    input_data = np.zeros(len(data_columns))

    input_data[0] = total_sqft
    input_data[1] = bhk
    input_data[2] = bath

    if location in location_columns:
        loc_index = location_columns.index(location) + 3
        input_data[loc_index] = 1
    else:
        st.error("Location not found. Please select a valid location.")

    try:
        predicted_price = model.predict([input_data])[0]
        st.success(f"The predicted price of the house is: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
