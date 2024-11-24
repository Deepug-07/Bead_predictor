import streamlit as st
from model import predict_outputs  # Ensure this import is correct

# Streamlit app layout
st.title("Model Prediction App")
st.write("Enter the parameters for prediction:")

# User input
wfs = st.number_input("WFS (e.g., 6.0)", min_value=0.0)
ts = st.number_input("TS (e.g., 150)", min_value=0.0)
voltage = st.number_input("Voltage (e.g., 19)", min_value=0.0)

if st.button("Predict"):
    predicted_output = predict_outputs(wfs, ts, voltage)
    st.success(f"Predicted Output: {predicted_output}")