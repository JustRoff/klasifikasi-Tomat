import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_tomat.joblib")
scaler = joblib.load("scaler_klasifikasi_tomat.joblib")

st.title("🍅klasifikasi Tomat")
st.markdown("Aplikasi machine learning untuk tomar termasuk kategori ** Ekspor,Industri,atau Lokal Premium **")
 
berat = st.slider("Berat Tomat: ", 50,200,80)
kekenyalan = st.slider("kekenyalan Tomat: ", 2.0,10.0,4.2)
kadar_gula = st.slider("Kadar Gula: ", 1.0,10.0,5.3)
tebal_kulit = st.slider("Berat Tomat:", 0.1,1.0,0.7)

if st.button("Prediksi"):
    data_baru = pd.DataFrame([[berat, kekenyalan, kadar_gula, tebal_kulit]],
                             columns=["berat","kekenyalan","kadar_gula","tebal_kulit"])
    
    data_baru_scaled = scaler.transform(data_baru)
    prediksi = model.predict(data_baru_scaled)[0]
    persentase = max(model.predict_proba(data_baru_scaled)[0])
    st.success(f"Model Memprediksi {prediksi} dengan keyakinan {persentase*100:.2f}%")
    st.balloons()

st.divider()
st.caption("Dibuat dengan 🍅 Oleh **Jr_**")