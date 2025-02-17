import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load the model
model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))

# Load the columns information
with open("columns.json", "r") as f:
    data_columns = json.load(f)

data_columns = data_columns['data_columns']
location_columns = data_columns[3:]

# Load the dataset containing home prices data
location_data = pd.read_csv("banglore_data.csv")  # Replace with your actual data source

st.title("Banglore Home Price Prediction")
st.header("Enter Home Details:")

# Dropdown for location selection
location = st.selectbox("Location", location_columns)

# Initialize minimum values
min_total_sqft = 300.0  # Default value for total_sqft
min_bhk_number = 1       # Default value for BHK
min_bathrooms = 1        # Default value for bathrooms

# Check if the selected location exists in the dataset
if location in location_data['location'].values:
    st.write(f"Selected location: {location}")
    # Filter data for the selected location
    location_filtered = location_data[location_data['location'] == location]
    
    # Sort the filtered DataFrame by 'total_sqft' in ascending order
    location_sorted = location_filtered.sort_values(by='total_sqft')
    
    # Get the row with the minimum total_sqft
    min_values = location_sorted.iloc[0]
    # Update minimum values based on the row with the minimum total_sqft
    min_total_sqft = min_values['total_sqft']  # Minimum square feet value
    min_bhk_number = min_values['bhk']          # Minimum BHK value
    min_bathrooms = min_values['bath']     # Minimum bathrooms value

# Input fields with dynamic min values
total_square_feet = st.number_input("Total Square Feet Area", min_value=min_total_sqft, step=0.1)
bhk_number = st.number_input("Number of Bedrooms (BHK)", min_value=min_bhk_number, step=1)
number_of_bathrooms = st.number_input("Number of Bathrooms", min_value=min_bathrooms, step=1)

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
