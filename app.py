import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- 1. DOSYALARI YÜKLEME ---
try:
    model = joblib.load('kalp_modeli.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    X_train_scaled = joblib.load('train_data.pkl') # SHAP için referans veri
except FileNotFoundError:
    st.error("Eksik dosya! Lütfen 'kalp_modeli.pkl', 'scaler.pkl', 'encoder.pkl' ve 'train_data.pkl' dosyalarını aynı klasöre koyun.")
    st.stop()

# --- 2. SAYFA AYARLARI ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# --- 3. PROFİLLER (SENARYOLAR) ---
high_risk_profile = {
    'age': 63, 'sex': 1, 'trestbps': 150, 'chol': 290, 'thalach': 115, 'oldpeak': 2.6,
    'cp': 3, 'fbs': 1, 'restecg': 2, 'exang': 1, 'slope': 1, 'ca': 3, 'thal': 2
}

# GÜNCELLENMİŞ ORTA RİSK
medium_risk_profile = {
    'age': 52, 'sex': 1, 'trestbps': 130, 'chol': 245, 'thalach': 155, 'oldpeak': 0.8,
    'cp': 1, 'fbs': 0, 'restecg': 1, 'exang': 0, 'slope': 1, 'ca': 0, 'thal': 1
}

low_risk_profile = {
    'age': 38, 'sex': 0, 'trestbps': 115, 'chol': 185, 'thalach': 175, 'oldpeak': 0.0,
    'cp': 2, 'fbs': 0, 'restecg': 0, 'exang': 0, 'slope': 0, 'ca': 0, 'thal': 0
}

default_values = {
    'age': 50, 'sex': 1, 'trestbps': 120, 'chol': 200, 'thalach': 150, 'oldpeak': 1.0,
    'cp': 0, 'fbs': 0, 'restecg': 0, 'exang': 0, 'slope': 0, 'ca': 0, 'thal': 0
}

def load_profile(profile):
    for key, value in profile.items():
        st.session_state[key] = value

if 'age' not in st.session_state:
    load_profile(default_values)

# --- 4. ARAYÜZ ---
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("Predict heart disease risk with **Explainable AI (XAI)**.")

st.subheader("📝 Quick Test Profiles")
b_col1, b_col2, b_col3 = st.columns(3)

with b_col1:
    if st.button("🚨 High Risk Patient", use_container_width=True):
        load_profile(high_risk_profile)
with b_col2:
    if st.button("⚠️ Suspicious Patient", use_container_width=True):
        load_profile(medium_risk_profile)
with b_col3:
    if st.button("✅ Healthy Person", use_container_width=True):
        load_profile(low_risk_profile)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Personal & Clinical")
    age = st.number_input("Age", min_value=1, max_value=120, key='age')
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key='sex')
    trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=250, key='trestbps')
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, key='chol')
    thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, key='thalach')
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, key='oldpeak')

with col2:
    st.subheader("Condition Details")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], key='cp',
                      format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x])
    fbs = st.selectbox("Fasting BS > 120?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key='fbs')
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2], key='restecg',
                           format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    exang = st.selectbox("Exercise Angina?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key='exang')
    slope = st.selectbox("ST Slope", options=[0, 1, 2], key='slope',
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3], key='ca')
    thal = st.selectbox("Thalassemia", options=[0, 1, 2], key='thal',
                        format_func=lambda x: ["Normal", "Fixed", "Reversable"][x])

# --- 5. TAHMİN VE AÇIKLANABİLİRLİK ---
st.divider()
if st.button("🔍 Analyze & Explain", type="primary", use_container_width=True):
    
    # DataFrame oluştur
    raw_data = pd.DataFrame({
        'age': [st.session_state.age], 'sex': [st.session_state.sex], 
        'trestbps': [st.session_state.trestbps], 'chol': [st.session_state.chol],
        'fbs': [st.session_state.fbs], 'thalach': [st.session_state.thalach], 
        'exang': [st.session_state.exang], 'oldpeak': [st.session_state.oldpeak],
        'cp': [st.session_state.cp], 'restecg': [st.session_state.restecg], 
        'slope': [st.session_state.slope], 'ca': [st.session_state.ca], 'thal': [st.session_state.thal]
    })

    # Veri İşleme (Pipeline)
    raw_data['trestbps'] = raw_data['trestbps'].clip(upper=170.0)
    raw_data['chol'] = raw_data['chol'].clip(lower=126.0, upper=374.5)
    raw_data['oldpeak'] = raw_data['oldpeak'].clip(upper=4.5)
    raw_data['thalach'] = raw_data['thalach'].clip(lower=84.75, upper=202.0)

    cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    num_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak']
    
    cat_encoded = encoder.transform(raw_data[cat_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
    
    processed_data = pd.concat([raw_data[num_cols], cat_df], axis=1)
    processed_data = processed_data[scaler.feature_names_in_]
    
    scaled_data = scaler.transform(processed_data)

    # Tahmin
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    # --- SONUÇ GÖSTERİMİ ---
    st.subheader("1. Prediction Result")
    col_res1, col_res2 = st.columns([3, 1])
    
    with col_res1:
        st.progress(int(probability * 100))
        if prediction[0] == 1:
            st.error(f"⚠️ **HIGH RISK** ({probability*100:.1f}%)")
        else:
            st.success(f"✅ **LOW RISK** ({probability*100:.1f}%)")

    # --- SHAP AÇIKLAMASI ---
  # --- SHAP AÇIKLAMASI (TÜM ÖZELLİKLER) ---
    st.subheader("2. Why this result? (Explainability)")
    with st.spinner("Calculating feature impacts..."):
        
        feature_names = processed_data.columns.tolist()
        toplam_ozellik_sayisi = len(feature_names) # Özellik sayısını otomatik alalım
        
        explainer = shap.LinearExplainer(model, X_train_scaled)
        shap_values = explainer(scaled_data)
        shap_values.feature_names = feature_names
        
        # Grafik boyutunu özellik sayısına göre dinamik ayarlayalım
        # Her özellik için dikeyde biraz yer açıyoruz (Minimum 8 birim, özellik başına 0.5 birim)
        dynamic_height = max(8, toplam_ozellik_sayisi * 0.5)
        fig, ax = plt.subplots(figsize=(10, dynamic_height))
        
        # max_display parametresini toplam sayıya eşitliyoruz
        shap.plots.waterfall(shap_values[0], max_display=toplam_ozellik_sayisi, show=False)
        
        plt.title(f"Impact of All {toplam_ozellik_sayisi} Features", fontsize=16)
        st.pyplot(fig)
        
    st.info("""
    **Graph Guide:**
    This waterfall plot shows **every single factor** considered by the model.
    * **Red bars (→):** Push the risk score UP.
    * **Blue bars (←):** Push the risk score DOWN.
    """)

