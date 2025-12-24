import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. DOSYALARI YÜKLEME ---
try:
    model = joblib.load('kalp_modeli.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
except FileNotFoundError:
    st.error("Required files not found! Please place 'kalp_modeli.pkl', 'scaler.pkl', and 'encoder.pkl' in the same folder.")
    st.stop()

# --- 2. SAYFA AYARLARI ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# --- 3. PROFİL VERİLERİ (SENARYOLAR) ---
# Burada senin belirlediğin 3 farklı hasta profilini tanımlıyoruz.
high_risk_profile = {
    'age': 63, 'sex': 1, 'trestbps': 150, 'chol': 290, 'thalach': 115, 'oldpeak': 2.6,
    'cp': 3, 'fbs': 1, 'restecg': 2, 'exang': 1, 'slope': 1, 'ca': 3, 'thal': 2
}

medium_risk_profile = {
    'age': 52, 
    'sex': 1, 
    'trestbps': 130, 
    'chol': 245, 
    'thalach': 155,       # Nabzı biraz iyileştirdik (145 -> 155)
    'oldpeak': 0.8,       # ST depresyonunu düşürdük (1.2 -> 0.8)
    'cp': 1,              # Atipik Anjina (Hala bir göğüs ağrısı var)
    'fbs': 0, 
    'restecg': 1,         # EKG'de hafif anormallik var
    'exang': 0,           # KRİTİK DEĞİŞİKLİK: Egzersizle ağrı yok (1 -> 0)
    'slope': 1,           # Düz eğim (Hala şüpheli)
    'ca': 0,              # KRİTİK DEĞİŞİKLİK: Damar tıkanıklığı yok (1 -> 0)
    'thal': 1             # Sabit Defekt (Şüpheli)
}
low_risk_profile = {
    'age': 38, 'sex': 0, 'trestbps': 115, 'chol': 185, 'thalach': 175, 'oldpeak': 0.0,
    'cp': 2, 'fbs': 0, 'restecg': 0, 'exang': 0, 'slope': 0, 'ca': 0, 'thal': 0
}

# Varsayılan değerler (Başlangıçta boş kalmaması için orta değerler)
default_values = {
    'age': 50, 'sex': 1, 'trestbps': 120, 'chol': 200, 'thalach': 150, 'oldpeak': 1.0,
    'cp': 0, 'fbs': 0, 'restecg': 0, 'exang': 0, 'slope': 0, 'ca': 0, 'thal': 0
}

# --- 4. SESSION STATE BAŞLATMA VE GÜNCELLEME ---
# Streamlit'te bir butona basınca formun değişmesi için bu fonksiyonu kullanıyoruz.
def load_profile(profile):
    for key, value in profile.items():
        st.session_state[key] = value

# Eğer session state henüz boşsa (ilk açılış), varsayılanları yükle
if 'age' not in st.session_state:
    load_profile(default_values)

# --- 5. ARAYÜZ ---
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("Predict heart disease risk using clinical data. You can select a quick profile below to test the system.")

# --- HIZLI PROFİL BUTONLARI ---
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

# --- FORM ALANI (Session State'e bağlı) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal & Clinical")
    # 'key' parametresi sayesinde bu inputlar session_state'e bağlanır ve butonla güncellenir.
    age = st.number_input("Age", min_value=1, max_value=120, key='age')
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key='sex')
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, key='trestbps')
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, key='chol')
    thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, key='thalach')
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, key='oldpeak')

with col2:
    st.subheader("Condition Details")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], key='cp',
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key='fbs')
    
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2], key='restecg',
                           format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    
    exang = st.selectbox("Exercise Induced Angina?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key='exang')
    
    slope = st.selectbox("ST Slope", options=[0, 1, 2], key='slope',
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    
    ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3], key='ca')
    
    thal = st.selectbox("Thalassemia", options=[0, 1, 2], key='thal',
                        format_func=lambda x: ["Normal", "Fixed Defect", "Reversable Defect"][x])

# --- 6. TAHMİN İŞLEMİ ---
st.divider()
if st.button("🔍 Analyze Result", type="primary", use_container_width=True):
    
    # Veriyi DataFrame'e çevir (Session State'ten gelen güncel değerlerle)
    raw_data = pd.DataFrame({
        'age': [st.session_state.age], 'sex': [st.session_state.sex], 
        'trestbps': [st.session_state.trestbps], 'chol': [st.session_state.chol],
        'fbs': [st.session_state.fbs], 'thalach': [st.session_state.thalach], 
        'exang': [st.session_state.exang], 'oldpeak': [st.session_state.oldpeak],
        'cp': [st.session_state.cp], 'restecg': [st.session_state.restecg], 
        'slope': [st.session_state.slope], 'ca': [st.session_state.ca], 'thal': [st.session_state.thal]
    })

    # --- Veri İşleme (Notebook Mantığıyla Birebir Aynı) ---
    
    # 1. Winsorization (Baskılama)
    raw_data['trestbps'] = raw_data['trestbps'].clip(upper=170.0)
    raw_data['chol'] = raw_data['chol'].clip(lower=126.0, upper=374.5)
    raw_data['oldpeak'] = raw_data['oldpeak'].clip(upper=4.5)
    raw_data['thalach'] = raw_data['thalach'].clip(lower=84.75, upper=202.0)

    # 2. Encoding
    cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    num_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak']
    
    cat_encoded = encoder.transform(raw_data[cat_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
    
    # 3. Birleştirme ve Sıralama
    processed_data = pd.concat([raw_data[num_cols], cat_df], axis=1)
    processed_data = processed_data[scaler.feature_names_in_]

    # 4. Scaling
    scaled_data = scaler.transform(processed_data)

    # 5. Tahmin
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    # --- SONUÇ GÖSTERİMİ ---
    st.subheader("Result")
    
    # Olasılık çubuğu (Progress bar)
    st.progress(int(probability * 100))
    
    if prediction[0] == 1:
        st.error(f"⚠️ **HIGH RISK** Detected")
        st.write(f"Patient has a **{probability*100:.1f}%** probability of heart disease.")
        st.info("Recommendation: Immediate clinical evaluation required.")
    else:
        st.success(f"✅ **LOW RISK** Detected")
        st.write(f"Patient has a **{probability*100:.1f}%** probability of heart disease.")
        st.info("Recommendation: Maintain healthy lifestyle.")