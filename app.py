import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('laptop_model.pkl', 'rb'))

st.title("💻 Pro Laptop Price Predictor")
st.subheader("Project by: Noor Ullah")

# Dropdowns for User Input
company = st.selectbox('Brand', ['Apple', 'HP', 'Dell', 'Lenovo', 'Asus', 'Acer'])
type_name = st.selectbox('Type', ['Netbook', 'Ultrabook', 'Notebook', 'Gaming', 'Workstation'])
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
cpu = st.selectbox('CPU Brand', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'Other'])
gpu = st.selectbox('GPU Brand', ['Intel', 'Nvidia', 'AMD'])

if st.button('Predict Price'):
    # In a real multiple regression, we would encode these strings to numbers
    # For now, let's show the logic
    # Note: You will need to retrain your model with these features in Colab first
    prediction = model.predict([[ram]]) # Baseline logic
    st.success(f"Estimated Price for {company} {type_name}: {prediction[0]:.2f} Euros")
