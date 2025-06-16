import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, dan label encoder
model = joblib.load('model_ann.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Prediksi Kategori Cuaca 🌦️ Berbasis ANN")

st.markdown("Masukkan parameter cuaca berikut:")

# Input form
DDD_CAR = st.number_input('DDD_CAR (Derajat angin terbanyak)', min_value=0.0, max_value=360.0, value=90.0)
TX = st.number_input('TX (Suhu maksimum)', min_value=20.0, max_value=50.0, value=32.0)
DDD_X = st.number_input('DDD_X (Arah angin maksimum)', min_value=0.0, max_value=360.0, value=180.0)
RH_AVG = st.number_input('RH_AVG (Kelembaban rata-rata)', min_value=0.0, max_value=100.0, value=80.0)
TAVG = st.number_input('TAVG (Suhu rata-rata)', min_value=20.0, max_value=40.0, value=28.0)
FF_AVG = st.number_input('FF_AVG (Kecepatan angin rata-rata)', min_value=0.0, max_value=15.0, value=5.0)

# Prediksi saat tombol ditekan
if st.button("Prediksi Cuaca"):
    input_data = pd.DataFrame([{
        'DDD_CAR': DDD_CAR,
        'TX': TX,
        'DDD_X': DDD_X,
        'RH_AVG': RH_AVG,
        'TAVG': TAVG,
        'FF_AVG': FF_AVG
    }])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)
    label = le.inverse_transform(prediction)[0]

    st.success(f"🌤️ Hasil Prediksi: **{label.upper()}**")
