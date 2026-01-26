import joblib
import streamlit as st
import pandas as pd
import numpy as np

model = joblib.load("laptop_price_predictor.joblib")
data = joblib.load("laptop_clean_data.joblib")

st.title("💻 Laptop Price Predictor")

company     = st.selectbox("Company", sorted(data['Company'].unique()))
typename    = st.selectbox("Type Name", sorted(data['TypeName'].unique()))
ram         = st.selectbox("RAM (GB)", sorted(data['Ram'].unique()))
weight      = st.selectbox("Weight (kg)", sorted(data['Weight'].unique()))
touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
ips_panel   = st.selectbox("IPS Panel", ['No', 'Yes'])
cpu_brand   = st.selectbox("CPU Brand", sorted(data['cpu brand'].unique()))
ssd         = st.selectbox("SSD (GB)", sorted(data['ssd'].unique()))
hdd         = st.selectbox("HDD (GB)", sorted(data['hdd'].unique()))
gpu_brand   = st.selectbox("GPU Brand", sorted(data['gpu brand'].unique()))
os          = st.selectbox("Operating System", sorted(data['os'].unique()))
ppi         = st.selectbox("PPI (Pixels Per Inch)", sorted(data['ppi'].unique()))





if st.button("Predict Price"):

    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val         = 1 if ips_panel == 'Yes' else 0

    input_df = pd.DataFrame([[company, typename, ram, weight,
                              touchscreen_val, ips_val, ppi,
                              cpu_brand, ssd, hdd, gpu_brand, os]],
                            columns=['Company','TypeName','Ram','Weight',
                                     'Touchscreen','IPS Panel','ppi',
                                     'cpu brand','ssd','hdd','gpu brand','os'])

    price_pred = model.predict(input_df)[0]
    ans = np.exp(price_pred)    
    st.success(f" Estimated Price: ₹{ans}")
    st.balloons()
