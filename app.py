import streamlit as st
import pickle
import numpy as np

# Load data and model
with open('laptop_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
le_company = data['le_company']
le_type = data['le_type']
le_cpu = data['le_cpu']
le_gpu = data['le_gpu']

st.title("💻 Laptop Price Predictor")
st.subheader("Project by: Noor Ullah")

# Dropdowns for user input
company = st.selectbox('Brand', le_company.classes_)
type_name = st.selectbox('Type', le_type.classes_)
ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
cpu = st.selectbox('CPU', le_cpu.classes_)
gpu = st.selectbox('GPU', le_gpu.classes_)

if st.button('Predict Price'):
    # Transform input text to numbers
    brand_encoded = le_company.transform([company])[0]
    type_encoded = le_type.transform([type_name])[0]
    cpu_encoded = le_cpu.transform([cpu])[0]
    gpu_encoded = le_gpu.transform([gpu])[0]
    
    query = np.array([[brand_encoded, type_encoded, ram, cpu_encoded, gpu_encoded]])
    prediction = model.predict(query)
    
    st.success(f"Estimated Price: {prediction[0]:.2f} Euros")
