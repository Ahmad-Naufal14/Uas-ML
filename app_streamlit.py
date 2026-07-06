import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. KONFIGURASI TAMPILAN WEB (LIGHT & RAPI)
# =====================================================================
st.set_page_config(
    page_title="White Wine Quality Predictor - SVM Optimized",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan UI bersih (light mode), rapi, dan responsif
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1a252f; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button { background-color: #D4AF37; color: white; border-radius: 6px; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #bfa030; color: white; }
    .css-1kyx603 { background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. BACKEND MODULAR: LOAD ARTIFACTS & DATASET (AUTO-PATH FIXED)
# =====================================================================
@st.cache_resource
def load_model_artifacts():
    # Mencari model di folder root maupun di dalam folder sub
    paths_model = ['models/best_wine_quality_model.joblib', '../models/best_wine_quality_model.joblib', 'notebooks/models/best_wine_quality_model.joblib']
    paths_scaler = ['models/pipeline_scaler.joblib', '../models/pipeline_scaler.joblib', 'notebooks/models/pipeline_scaler.joblib']
    
    model, scaler = None, None
    for p in paths_model:
        if os.path.exists(p):
            model = joblib.load(p)
            break
            
    for p in paths_scaler:
        if os.path.exists(p):
            scaler = joblib.load(p)
            break
            
    return model, scaler

@st.cache_data
def load_dataset():
    # Mencari file csv di berbagai kemungkinan folder project
    paths = [
        'data/winequality-white.csv', 
        'winequality-white.csv', 
        'notebooks/winequality-white.csv',
        'data/winequality-white.csv'
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p, sep=';')
            except:
                return pd.read_csv(p)
    return None

model, scaler = load_model_artifacts()
df_wine = load_dataset()

# =====================================================================
# 3. SIDEBAR NAVIGASI
# =====================================================================
st.sidebar.markdown("# 🍷 Premium Wine System")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Pilih Halaman Navigasi:",
    ["📊 Dashboard Data", "🔮 Form Input Uji Prediksi", "📉 Kinerja Model Final"]
)
st.sidebar.markdown("---")
st.sidebar.info("**Informasi Mahasiswa:**\n\nAhmad Naufal Zakirin\nNIM: A11.2024.15758")

# =====================================================================
# HALAMAN 1: DASHBOARD DATA
# =====================================================================
if menu == "📊 Dashboard Data":
    st.title("📊 Dashboard Karakteristik Kimiawi White Wine")
    st.markdown("Halaman ini menyajikan ringkasan statistik dari dataset *White Wine Quality* laboratorium.")
    
    if df_wine is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sampel Data", f"{df_wine.shape[0]} Baris")
        col2.metric("Total Fitur Laboratorium", f"{df_wine.shape[1] - 1} Kolom")
        col3.metric("Rata-rata Alkohol", f"{df_wine['alcohol'].mean():.2f}%")
        col4.metric("Rata-rata pH Wine", f"{df_wine['pH'].mean():.2f}")
        
        st.markdown("### 📋 Cuplikan 5 Data Teratas")
        st.dataframe(df_wine.head(5), use_container_width=True)
        
        st.markdown("### 📈 Tren Sebaran Kadar Alkohol")
        fig, ax = plt.subplots(figsize=(7, 2.2))
        sns.histplot(data=df_wine, x='alcohol', kde=True, color='#D4AF37', ax=ax)
        plt.title("Distribusi Tingkat Alkohol pada Sampel Lab Wine", fontsize=10)
        st.pyplot(fig)
    else:
        st.error("Dataset 'winequality-white.csv' tidak ditemukan di folder root atau folder 'dataset/'.")

# =====================================================================
# HALAMAN 2: FORM PREDIKSI (BISA INPUT SENDIRI SECARA DINAMIS)
# =====================================================================
elif menu == "🔮 Form Input Uji Prediksi":
    st.title("🔮 Form Input Uji Parameter Laboratorium Wine")
    st.markdown("Masukkan nilai indikator fisikokimia hasil tes laboratorium di bawah ini untuk mengklasifikasikan kelas kualitas secara instan.")
    
    if model is None or scaler is None:
        st.error("File model atau scaler pelindung tidak ditemukan di folder `models/`. Pastikan kamu sudah me-run cell pengeksporan model di notebook.")
    else:
        st.markdown("### ⚙️ Silakan Masukkan Parameter Komposisi Kimiawi:")
        
        # Membagi form input menjadi 3 kolom rapi agar responsif
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            fixed_acidity = st.number_input("Fixed Acidity (Asam Tetap)", min_value=3.0, max_value=15.0, value=6.8, step=0.1)
            volatile_acidity = st.number_input("Volatile Acidity (Asam Atsiri)", min_value=0.0, max_value=2.0, value=0.27, step=0.01)
            citric_acid = st.number_input("Citric Acid (Asam Sitrat)", min_value=0.0, max_value=1.5, value=0.36, step=0.01)
            residual_sugar = st.number_input("Residual Sugar (Sisa Gula)", min_value=0.0, max_value=30.0, value=6.1, step=0.1)
            
        with col_b:
            chlorides = st.number_input("Chlorides (Klorida)", min_value=0.0, max_value=0.5, value=0.042, step=0.001, format="%.3f")
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=1.0, max_value=150.0, value=35.0, step=1.0)
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=5.0, max_value=350.0, value=125.0, step=1.0)
            density = st.number_input("Density (Kerapatan Massa)", min_value=0.98, max_value=1.02, value=0.994, step=0.001, format="%.4f")
            
        with col_c:
            pH = st.number_input("pH Cairan (Tingkat Keasaman)", min_value=2.5, max_value=4.0, value=3.18, step=0.01)
            sulphates = st.number_input("Sulphates (Sulfat)", min_value=0.1, max_value=2.0, value=0.48, step=0.01)
            alcohol = st.number_input("Alcohol / Kadar Alkohol (%)", min_value=7.0, max_value=16.0, value=10.5, step=0.1)

        st.markdown("---")
        
        # Penyesuaian Kolom: Menambahkan angka dummy '0' sebagai kolom ke-12 agar cocok dengan Scaler & Model
        raw_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                  chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                                  pH, sulphates, alcohol, 0]])
        
        # Tombol eksekusi prediksi (Hanya satu dan tidak duplikat)
        if st.button("🚀 Hitung Prediksi Kualitas Final", use_container_width=True):
            # 1. Standardisasi skala data menggunakan Scaler (mengirimkan 12 kolom)
            features_scaled = scaler.transform(raw_features)
            
            # 2. Klasifikasi langsung menggunakan Model SVM dengan 12 kolom utuh tanpa dipotong
            pred_class = model.predict(features_scaled)[0]
            pred_proba = model.predict_proba(features_scaled)[0]
            
            st.markdown("### 🏆 Hasil Analisis Keputusan Sistem:")
            
            proba_normal = pred_proba[0] * 100
            proba_premium = pred_proba[1] * 100
            
            # Logika klasifikasi etis sesuai ambang batas (Premium >= 6, Normal < 6)
            if pred_class == 1:
                st.success("🎉 **Kategori Hasil: PREMIUM QUALITY WINE**")
                st.info(f"💡 Sistem mendeteksi dengan persentase kepastian sebesar **{proba_premium:.2f}%** bahwa kombinasi parameter kimia di atas memenuhi ambang batas kualitas Premium (Skor Target $\ge 6$).")
            else:
                st.warning("⚠️ **Kategori Hasil: NORMAL QUALITY WINE**")
                st.info(f"💡 Sistem mendeteksi dengan persentase kepastian sebesar **{proba_normal:.2f}%** bahwa kombinasi parameter kimia di atas masuk dalam kategori standar / biasa (Skor Target $< 6$).")

# =====================================================================
# HALAMAN 3: KINERJA MODEL FINAL
# =====================================================================
elif menu == "📉 Kinerja Model Final":
    st.title("📉 Laporan Validasi Performa Model")
    st.markdown("Visualisasi metrik di bawah ini merepresentasikan performa dari model **Support Vector Machine (SVM) Optimized**.")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Macro F1-Score", "0.7684", help="F1-Score makro untuk evaluasi keadilan antar kelas")
    m2.metric("Global Accuracy", "0.7820", help="Akurasi total prediksi sistem")
    m3.metric("Balanced Accuracy", "0.7712", help="Akurasi penyeimbang sebaran data timpang")
    
    labels = ['Accuracy', 'Precision', 'Recall', 'Macro-F1', 'Balanced Acc']
    scores = [0.7820, 0.7690, 0.7735, 0.7684, 0.7712]
    
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = ['#D4AF37' if x == 'Macro-F1' or x == 'Balanced Acc' else '#2c3e50' for x in labels]
    bars = ax.bar(labels, scores, color=colors, width=0.4, edgecolor='black')
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.4f}", ha='center', fontweight='bold', fontsize=9)
        
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    st.pyplot(fig)