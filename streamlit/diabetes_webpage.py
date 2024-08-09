# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:22:45 2024

@author: sindh
"""

import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
loaded_model = pickle.load(open('C:/Users/sindh/Downloads/GoldPrice_prediction_Docker/gld_model.sav', 'rb'))

def gold_prediction(input_data):
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make the prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    return prediction[0]

def main():
    st.title("Gold Price Prediction Web Page")
    
    # Get user input
    SPX = st.text_input("SPX")
    USO = st.text_input("USO")
    SLV = st.text_input("SLV")
    EUR_USD = st.text_input("EUR/USD")
    
    predictions = ''
    
    if st.button("Predict Gold Price"):
        # Ensure all inputs are provided
        if SPX and USO and SLV and EUR_USD:
            try:
                # Convert input values to float and predict
                predictions = gold_prediction([float(SPX), float(USO), float(SLV), float(EUR_USD)])
                st.success(f'Predicted Gold Price: {predictions}')
            except ValueError:
                st.error("Please enter valid numeric values.")
        else:
            st.error("Please enter values for all fields.")
    
if __name__ == '__main__':
    main()
