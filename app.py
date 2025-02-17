import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("bengaluru_house_prices.csv")  # Replace with your actual dataset

# Load the trained model
model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))

# Load column names
with open("columns.json", "r") as f:
    data_columns = json.load(f)

data_columns = data_columns['data_columns']
location_columns = data_columns[3:]

st.title("Bangalore Home Price Prediction")

st.header("Enter Home Details:")

# Select location
location = st.selectbox("Location", location_columns)

# Filter data for the selected location
location_data = df[df['location'] == location]

# Get minimum values for the selected location
if not location_data.empty:
    min_sqft = location_data['total_sqft'].min()
    min_bhk = location_data['bhk'].min()
    min_bathrooms = location_data['bathrooms'].min()
# else:
#     min_sqft = 300.0  # Default min value
#     min_bhk = 1
#     min_bathrooms = 1

# Input fields with dynamic min values
total_square_feet = st.number_input("Total Square Feet Area", min_value=float(min_sqft), step=0.1)
bhk_number = st.number_input("Number of Bedrooms (BHK)", min_value=int(min_bhk), step=1)
number_of_bathrooms = st.number_input("Number of Bathrooms", min_value=int(min_bathrooms), step=1)

if st.button("Predict"):
    input_data = np.zeros(len(data_columns))

    input_data[0] = total_square_feet
    input_data[1] = bhk_number
    input_data[2] = number_of_bathrooms

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
