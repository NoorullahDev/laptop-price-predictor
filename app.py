
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('laptop_model.pkl', 'rb'))

st.title("💻 Laptop Price Predictor")
st.write("Project by: Noor Ullah")

ram = st.number_input("Enter RAM (GB)", min_value=2, max_value=64, value=8)

if st.button("Predict Price"):
    prediction = model.predict([[ram]])
    st.success(f"Estimated Price: {prediction[0]:.2f} Euros")
