import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image
import shap
import matplotlib.pyplot as plt
from src.services.llm_advisor import get_lifestyle_recommendation


# Fungsi untuk memuat model dan scaler (dicache agar lebih cepat)
@st.cache_resource
def load_artifacts():
    """
    Fungsi ini memuat model dan scaler dari file pickle.
    Menggunakan st.cache_resource untuk memastikan file hanya dimuat sekali.
    """
    try:
        with open('heart_disease_artifacts.pkl', 'rb') as file:
            artifacts = pickle.load(file)
        return artifacts['best_model'], artifacts['scaler']
    except FileNotFoundError:
        return None, None

# Memuat artifak (model dan scaler)
model, scaler = load_artifacts()

# Fungsi untuk melakukan prediksi
def predict_heart_disease(input_data):
    """
    Fungsi ini mengambil input, melakukan scaling, dan mengembalikan hasil prediksi.
    """
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    return prediction[0]

# --- Definisi Halaman-Halaman Aplikasi ---

def page_home():
    st.title("Selamat Datang di Aplikasi Prediksi Penyakit Jantung")
    st.markdown("""
    Aplikasi ini menggunakan model machine learning untuk memprediksi kemungkinan seseorang menderita penyakit jantung berdasarkan data medis.
    Silakan gunakan sidebar untuk navigasi ke halaman lainnya.
    """)
    try:
        image = Image.open("heart.jpg")
        st.image(image)
    except FileNotFoundError:
        st.warning("File gambar tidak ditemukan. Pastikan file berada di folder yang sama.")

def page_prediksi():
    st.title("Prediksi Penyakit Jantung")
    st.markdown("Masukkan data pasien di bawah ini untuk melakukan prediksi.")
    st.divider()

    if model is None or scaler is None:
        st.error("Model atau scaler tidak ditemukan. Pastikan file `heart_disease_artifacts.pkl` ada di direktori yang sama.")
        return

    # --- Tata letak input dengan nama fitur medis dan tooltip penjelasan ---
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('age (Usia)', 29, 77, 54, help="Usia pasien dalam tahun.")
        with col2:
            thalach = st.slider('thalach (Detak Jantung Maksimal)', 71, 202, 150, help="Detak jantung tertinggi yang tercatat selama tes latihan.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            sex_options = {'Pria': 1, 'Wanita': 0}
            sex = st.radio('sex (Jenis Kelamin)', list(sex_options.keys()), horizontal=True, help="Jenis kelamin pasien.")
        with col2:
            oldpeak = st.slider('oldpeak (ST Depression)', 0.0, 6.2, 1.0, help="Penurunan segmen ST pada EKG saat latihan dibandingkan istirahat.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            cp_options = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
            cp = st.selectbox('cp (Jenis Nyeri Dada)', list(cp_options.keys()), help="Jenis nyeri dada yang dialami pasien.")
        with col2:
            slope_options = {'Upsloping': 2, 'Flat': 1, 'Downsloping': 0}
            slope = st.selectbox('slope (Kemiringan ST)', list(slope_options.keys()), help="Bentuk kemiringan segmen ST pada EKG saat puncak latihan.")
            
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            exang_options = {'Ya': 1, 'Tidak': 0}
            exang = st.selectbox('exang (Nyeri Dada Saat Olahraga)', list(exang_options.keys()), help="Apakah nyeri dada muncul atau memburuk saat berolahraga?")
        with col2:
            ca_options = {'0': 0, '1': 1, '2': 2, '3': 3}
            ca = st.selectbox('ca (Pembuluh Darah Utama)', list(ca_options.keys()), help="Jumlah pembuluh darah utama (0-3) yang terdeteksi memiliki penyempitan signifikan.")

    with st.container():
        _, col_centered, _ = st.columns([1, 2, 1])
        with col_centered:
             thal_options = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
             thal = st.selectbox('thal (Tes Thalassemia)', list(thal_options.keys()), index=1, help="Hasil tes darah Thalassemia, sebuah kelainan darah.")
        
    st.divider()

    # Tombol prediksi
    if st.button('Prediksi Penyakit Jantung', type="primary", use_container_width=True):
        input_data = [
            cp_options[cp], thalach, slope_options[slope], oldpeak,
            exang_options[exang], ca_options[ca], thal_options[thal],
            sex_options[sex], age
        ]
        with st.spinner('Melakukan prediksi...'):
            time.sleep(1)
            prediction = predict_heart_disease(input_data)
        st.subheader("Hasil Prediksi")
        if prediction == 1:
            st.error('**Pasien diprediksi menderita penyakit jantung.**', icon="💔")
        else:
            st.success('**Pasien diprediksi tidak menderita penyakit jantung.**', icon="❤️")
            
        st.divider()
        
        # --- SHAP XAI ---
        st.subheader("Analisis Faktor Risiko (Explainable AI)")
        st.write("Grafik di bawah ini menunjukkan faktor mana yang paling berkontribusi terhadap hasil prediksi (semakin ke kanan/merah semakin berisiko).")
        try:
            explainer = shap.TreeExplainer(model)
            std_data = scaler.transform(np.asarray(input_data).reshape(1, -1))
            shap_values = explainer.shap_values(std_data)
            
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]
            sv = np.ravel(sv)
                
            feature_names = ['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age']
            fig, ax = plt.subplots(figsize=(8, 4))
            sorted_idx = np.argsort(sv)
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_sv = sv[sorted_idx]
            colors = ['#ff4b4b' if v > 0 else '#32cd32' for v in sorted_sv]
            
            ax.barh(sorted_features, sorted_sv, color=colors)
            ax.set_title('Kontribusi Setiap Fitur Medis terhadap Prediksi')
            ax.set_xlabel('SHAP Value (Risiko)')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Gagal memuat visualisasi XAI: {e}")
            
        st.divider()
        
        # --- GenAI ---
        st.subheader("Saran Gaya Hidup (Generative AI)")
        with st.spinner("AI sedang menganalisis saran khusus untuk Anda..."):
            patient_data_dict = {
                'age': age,
                'sex': sex,
                'thalach': thalach,
                'cp': cp
            }
            recommendation = get_lifestyle_recommendation(patient_data_dict, prediction)
            st.info(recommendation)

def page_detail_proyek():
    st.title("Detail dan Metodologi Proyek")
    st.markdown("Halaman ini menjelaskan alur kerja teknis dalam pengembangan model prediksi, dari data mentah hingga produk jadi.")
    st.divider()

    st.header("1. Tentang Dataset")
    st.markdown("""
    Dataset yang digunakan adalah **Heart Disease** dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease). Dataset ini terdiri dari **1025 baris data** dan 13 atribut + 1 target.
    
    Kolom `target` menjadi acuan utama, di mana nilai **0** berarti tidak ada penyakit jantung dan **1** berarti terdeteksi adanya penyakit jantung.
    
    Berikut adalah rincian dari setiap atribut:
    - **age**: Usia pasien (tahun).
    - **sex**: Jenis kelamin (1 = Pria, 0 = Wanita).
    - **cp**: Jenis nyeri dada.
    - **trestbps**: Tekanan darah saat istirahat (mmHg).
    - **chol**: Kadar kolesterol serum (mg/dl).
    - **fbs**: Gula darah puasa > 120 mg/dl (1 = Ya, 0 = Tidak).
    - **restecg**: Hasil EKG saat istirahat.
    - **thalach**: Detak jantung maksimum yang tercapai.
    - **exang**: Nyeri dada akibat olahraga (1 = Ya, 0 = Tidak).
    - **oldpeak**: Depresi segmen ST pada EKG.
    - **slope**: Kemiringan segmen ST saat puncak latihan.
    - **ca**: Jumlah pembuluh darah utama yang terdeteksi (0-3).
    - **thal**: Hasil tes *thallium scan*.
    """)
    st.divider()

    st.header("2. Persiapan dan Eksplorasi Data")
    st.markdown("""
    - **Insight dari EDA**: Ditemukan bahwa dataset cukup seimbang antara kelas positif dan negatif. Analisis distribusi berhasil mengidentifikasi distribusi fitur-fitur dataset dan Analisis korelasi juga berhasil mengidentifikasi fitur-fitur yang paling berpengaruh terhadap target.
    - **Pembersihan Data**: Menghapus lebih dari **700 data duplikat** dan menangani nilai anomali.
    - **Penanganan Outlier**: Menghapus data pencilan pada fitur `trestbps` dan `chol`.
    - **Seleksi Fitur**: Memilih **9 fitur paling relevan** untuk difokuskan dalam pemodelan.
    - **Scaling & Pembagian Data**: `StandardScaler` diterapkan pada data, lalu dataset dibagi menjadi **80% data latih** dan **20% data uji**.
    """)
    st.divider()

    st.header("3. Pelatihan dan Pemilihan Model")
    st.markdown("""
    - **Pelatihan & Tuning**: Beberapa model kandidat (termasuk Logistic Regression, Decision Tree, dan Random Forest) dilatih dan dioptimalkan menggunakan `GridSearchCV`.
    - **Pemilihan Model Final**: **Random Forest** terpilih sebagai model final karena memberikan hasil evaluasi terbaik setelah proses tuning.
    """)
    
    st.subheader("Rincian Performa Model Terpilih (Random Forest)")
    st.markdown("Berikut adalah rincian performa model pada data uji:")
    st.markdown("""
    - **Kelas 0 (Negatif/Tidak Sakit Jantung)**:
        - **Precision: 0.87**
        - **Recall: 0.77**
        - **F1-Score: 0.82**
    - **Kelas 1 (Positif/Sakit Jantung)**:
        - **Precision: 0.82**
        - **Recall: 0.90**
        - **F1-Score: 0.86**
    """)
    st.markdown("- **Akurasi Keseluruhan**: 84%\n- **AUC-ROC**: 0.91")
    st.divider()

    st.header("4. Deployment")
    st.markdown("""
    - **Penyimpanan Artefak**: Model Random Forest terbaik dan scaler disimpan dalam satu file `.pkl` menggunakan `pickle`.
    - **Antarmuka Pengguna**: Streamlit digunakan untuk membangun aplikasi web ini, memungkinkan interaksi langsung dengan model yang telah dilatih.
    """)

def page_tentang_saya():
    st.title("Tentang Saya")
    st.markdown("""
    Halo! Saya **Arfi**, seorang mahasiswa Data Science dari Politeknik Elektronika Negeri Surabaya.
    Saya bersemangat mengeksplorasi bidang Data Science dan Artificial Intelligence untuk menciptakan solusi yang berdampak positif.

    Anda dapat terhubung dengan saya melalui:
    - **Email**: arfiadi1com
    - **LinkedIn**: [Arfi Adi Nugroho](https://www.linkedin.com/in/arfi-adi-nugroho-58b0a7263)
    - **GitHub**: [Arfiadi](https://github.com/Arfiadi)
    """)

# --- Navigasi dan Eksekusi Utama ---
st.sidebar.title("Navigasi")
page_options = {
    "Home": page_home,
    "Prediksi": page_prediksi,
    "Detail Proyek": page_detail_proyek,
    "Tentang Saya": page_tentang_saya
}
selected_page = st.sidebar.radio("Pilih Halaman:", list(page_options.keys()))

# Menjalankan fungsi halaman yang dipilih
page_options[selected_page]()