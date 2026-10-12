import os
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import shap

from src.services.llm_advisor import get_lifestyle_recommendation

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Artifact Loading ---
@st.cache_resource
def load_artifacts():
    """
    Load serialized model, scaler, and evaluation metrics.
    """
    try:
        with open('heart_disease_artifacts.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

artifacts = load_artifacts()
model = artifacts['best_model'] if artifacts else None
scaler = artifacts['scaler'] if artifacts else None

# Helper prediction function
def predict_heart_disease(input_data):
    """
    Predict heart disease (0 or 1) given 9 medical features.
    Features order: cp, thalach, slope, oldpeak, exang, ca, thal, sex, age
    """
    input_array = np.asarray(input_data).reshape(1, -1)
    scaled_data = scaler.transform(input_array)
    prediction = model.predict(scaled_data)
    return int(prediction[0])

def predict_heart_disease_proba(input_data):
    """
    Predict probability of heart disease given 9 medical features.
    """
    input_array = np.asarray(input_data).reshape(1, -1)
    scaled_data = scaler.transform(input_array)
    return model.predict_proba(scaled_data)[0, 1]

@st.cache_data(show_spinner=False)
def cached_lifestyle_recommendation(age: int, sex: int, thalach: int, cp: str, prediction: int) -> str:
    """
    Cached AI / clinical recommendation to avoid redundant API calls and prevent timeouts.
    """
    patient_data_dict = {
        'age': age,
        'sex': sex,
        'thalach': thalach,
        'cp': cp
    }
    return get_lifestyle_recommendation(patient_data_dict, prediction)

# --- Pages ---

def page_home():
    st.title("❤️ Sistem Prediksi Risiko Penyakit Jantung")
    st.markdown("""
    Sistem berbasis Machine Learning end-to-end yang dirancang untuk membantu tenaga medis 
    dan pasien dalam melakukan **skrining dini risiko kardiovaskular**.
    
    Aplikasi ini mengintegrasikan **Machine Learning (Random Forest)**, **Explainable AI (SHAP)**, 
    dan **Generative AI (Gemini)** untuk memberikan diagnosis transparan serta rekomendasi gaya hidup yang dipersonalisasi.
    """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Keunggulan Sistem")
        st.markdown("""
        - 📊 **Evaluasi Rigor & Cross-Validation**: Dilatih dengan *5-Fold Stratified Cross-Validation* untuk mencegah *overfitting* atau *lucky split*.
        - 🔍 **Explainable AI (XAI)**: Bukan model *black-box*; kontribusi setiap faktor risiko pasien divisualisasikan dengan SHAP values.
        - 🤖 **AI Lifestyle Advisor**: Dilengkapi rekomendasi gaya hidup otomatis dengan *graceful clinical fallback*.
        - ⚙️ **Standar MLOps**: Didukung tracking eksperimen MLflow, otomatisasi CI/CD, dan siap dikontainerisasi dengan Docker.
        """)
        st.info("💡 **Gunakan menu navigasi di sidebar** untuk mencoba formulir prediksi atau melihat analisis eksplorasi data (EDA).")
        
    with col2:
        try:
            image = Image.open("heart.jpg")
            st.image(image, caption="Sistem Diagnosis & Skrining Kardiovaskular", use_container_width=True)
        except Exception:
            st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=800", caption="Cardiovascular Care", use_container_width=True)


def page_prediksi():
    st.title("🩺 Form Skrining & Prediksi Risiko Pasien")
    st.markdown("Masukkan parameter rekam medis pasien di bawah ini untuk mendapatkan estimasi risiko dan rekomendasi medis.")
    st.divider()

    if model is None or scaler is None:
        st.error("Artefak model tidak ditemukan. Pastikan file `heart_disease_artifacts.pkl` tersedia.")
        return

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider('Usia (age)', 29, 77, 54, help="Usia pasien dalam satuan tahun.")
            sex_options = {'Pria': 1, 'Wanita': 0}
            sex = st.radio('Jenis Kelamin (sex)', list(sex_options.keys()), horizontal=True)
            cp_options = {
                'Typical Angina (Nyeri Dada Tipikal)': 0,
                'Atypical Angina (Nyeri Dada Atipikal)': 1,
                'Non-anginal Pain (Bukan Angina)': 2,
                'Asymptomatic (Tanpa Gejala Nyeri)': 3
            }
            cp = st.selectbox('Jenis Nyeri Dada (cp)', list(cp_options.keys()), help="Karakteristik nyeri dada yang dirasakan pasien.")
            
        with col2:
            thalach = st.slider('Detak Jantung Maksimal (thalach)', 71, 202, 150, help="Detak jantung tertinggi saat tes stres fisik/treadmill.")
            oldpeak = st.slider('Depresi ST EKG (oldpeak)', 0.0, 6.2, 1.0, step=0.1, help="Penurunan segmen ST saat latihan dibanding istirahat.")
            slope_options = {'Upsloping (Menaik)': 2, 'Flat (Datar)': 1, 'Downsloping (Menurun)': 0}
            slope = st.selectbox('Kemiringan Segmen ST (slope)', list(slope_options.keys()), help="Bentuk kemiringan segmen ST pada elektrokardiogram (EKG).")

        with col3:
            exang_options = {'Ya': 1, 'Tidak': 0}
            exang = st.selectbox('Nyeri Dada Saat Olahraga (exang)', list(exang_options.keys()), help="Apakah nyeri dada timbul akibat aktivitas fisik?")
            ca_options = {'0': 0, '1': 1, '2': 2, '3': 3}
            ca = st.selectbox('Jumlah Pembuluh Darah Utama (ca)', list(ca_options.keys()), help="Jumlah pembuluh darah utama (0-3) dengan penyempitan >50% fluoroskopi.")
            thal_options = {'Normal': 1, 'Fixed Defect (Cacat Permanen)': 2, 'Reversible Defect (Cacat Reversibel)': 3}
            thal = st.selectbox('Hasil Thallium Scan (thal)', list(thal_options.keys()), index=1, help="Hasil tes perfusi miokard Thallium.")

    st.divider()

    if st.button('Mulai Analisis & Prediksi', type="primary", use_container_width=True):
        input_data = [
            cp_options[cp], thalach, slope_options[slope], oldpeak,
            exang_options[exang], ca_options[ca], thal_options[thal],
            sex_options[sex], age
        ]
        
        with st.spinner('Menganalisis profil data medis pasien...'):
            time.sleep(0.5)
            prediction = predict_heart_disease(input_data)
            probability = predict_heart_disease_proba(input_data)

        st.subheader("📋 Hasil Diagnosis Model")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prediction == 1:
                st.error(f"### 💔 Terdeteksi Risiko Tinggi Penyakit Jantung\n**Probabilitas Risiko: {probability * 100:.1f}%**", icon="⚠️")
                st.markdown("Model mengidentifikasi indikator klinis yang mengarah pada kelainan kardiovaskular.")
            else:
                st.success(f"### ❤️ Risiko Rendah / Normal\n**Probabilitas Risiko: {probability * 100:.1f}%**", icon="✅")
                st.markdown("Parameter klinis pasien saat ini berada dalam rentang toleransi normal.")

        with res_col2:
            st.metric(label="Tingkat Keyakinan Model", value=f"{max(probability, 1 - probability) * 100:.1f}%")
            st.caption("Klasifikasi dihasilkan oleh Random Forest Classifier yang dioptimasi.")

        st.divider()

        # --- Explainable AI (SHAP) ---
        st.subheader("🔍 Explainable AI (XAI): Analisis Faktor Risiko Pasien (SHAP)")
        st.write("Grafik di bawah ini membedah faktor mana yang paling mendorong atau menurunkan risiko pada pasien ini (warna merah menunjukkan peningkatan risiko).")
        
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
                
            feature_names = ['cp (Nyeri Dada)', 'thalach (Detak Max)', 'slope (ST Slope)', 
                             'oldpeak (ST Dep)', 'exang (Angina)', 'ca (Pembuluh)', 
                             'thal (Thalassemia)', 'sex (Gender)', 'age (Usia)']
            
            fig, ax = plt.subplots(figsize=(9, 4.2))
            sorted_idx = np.argsort(sv)
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_sv = sv[sorted_idx]
            colors = ['#e63946' if v > 0 else '#2a9d8f' for v in sorted_sv]
            
            ax.barh(sorted_features, sorted_sv, color=colors)
            ax.set_title('Kontribusi Relatif Fitur terhadap Keputusan Prediksi (SHAP Value)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Pengaruh terhadap Risiko (+ Meningkatkan Risiko, - Menurunkan Risiko)')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Visualisasi SHAP tidak dapat dirender: {e}")

        st.divider()

        # --- Generative AI & Clinical Fallback ---
        st.subheader("💡 Rekomendasi Medis & Gaya Hidup Berbasis AI")
        with st.spinner("Menyusun rekomendasi gaya hidup yang dipersonalisasi..."):
            rec_text = cached_lifestyle_recommendation(age, sex_options[sex], thalach, cp, prediction)
            st.info(rec_text)


def page_eda_dan_evaluasi():
    st.title("📊 Eksplorasi Data (EDA) & Evaluasi Performa Model")
    st.markdown("Halaman ini menyajikan transparansi data science, metrik evaluasi model, dan wawasan klinis dari dataset.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🏆 Performa & Perbandingan Model", "📈 Feature Importance (MDI vs SHAP)", "🔬 Insight Eksplorasi Data (EDA)"])

    with tab1:
        st.subheader("1. Perbandingan Model Kandidat (Cross-Validation & Test Set)")
        st.markdown("""
        Evaluasi dilakukan menggunakan **5-Fold Stratified Cross-Validation** pada data latih serta pengujian independen pada *holdout test set* (20%).
        Seluruh percobaan dilacak secara otomatis menggunakan **MLflow**.
        """)
        
        if artifacts and "model_comparison" in artifacts:
            df_comp = artifacts["model_comparison"]
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        else:
            st.info("Data perbandingan model belum dimuat dari artefak.")

        st.divider()

        st.subheader("2. Evaluasi Diagnostik: Confusion Matrix & Kurva ROC")
        col_cm, col_roc = st.columns(2)

        if artifacts and "test_eval" in artifacts:
            eval_data = artifacts["test_eval"]
            cm = eval_data["confusion_matrix"]
            
            with col_cm:
                st.markdown("**Confusion Matrix (Holdout Test Set)**")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Prediksi Sehat (0)', 'Prediksi Sakit (1)'],
                            yticklabels=['Aktual Sehat (0)', 'Aktual Sakit (1)'], ax=ax_cm)
                ax_cm.set_ylabel('Status Aktual Pasien')
                ax_cm.set_xlabel('Hasil Prediksi Model')
                plt.tight_layout()
                st.pyplot(fig_cm)
                st.caption("📌 **Catatan Klinis**: Dalam domain kardiovaskular, meminimalkan **False Negative** (pasien sakit yang lolos diagnosis) adalah prioritas tertinggi guna mencegah mortalitas.")

            with col_roc:
                st.markdown(f"**Receiver Operating Characteristic (ROC Curve) - AUC: {eval_data.get('roc_auc', 0.898):.3f}**")
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(eval_data["fpr"], eval_data["tpr"], color='#1d3557', lw=2.5,
                            label=f"Random Forest (AUC = {eval_data.get('roc_auc', 0.898):.3f})")
                ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Baseline')
                ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
                ax_roc.set_ylabel('True Positive Rate (Recall / Sensitivity)')
                ax_roc.set_title('Kurva ROC Model Terpilih')
                ax_roc.legend(loc='lower right')
                ax_roc.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()
                st.pyplot(fig_roc)
                st.caption("Nilai AUC mendekati 0.90 menunjukkan kemampuan diskriminasi kelas yang sangat baik.")

    with tab2:
        st.subheader("Analisis Pentingnya Fitur (Random Forest Gini vs SHAP)")
        st.markdown("""
        Perbandingan dua pendekatan Explainability:
        - **Mean Decrease in Impurity (Gini Importance)**: Menilai seberapa sering fitur digunakan untuk memisahkan node pada seluruh pohon decision tree.
        - **SHAP (Shapley Additive exPlanations)**: Menilai kontribusi marginal fitur berbasis *game theory*.
        """)
        
        if artifacts and "feature_importance" in artifacts:
            df_feat = artifacts["feature_importance"]
            fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
            sns.barplot(data=df_feat, x="Importance", y="Feature", palette="viridis", ax=ax_fi)
            ax_fi.set_title("Random Forest Feature Importance (Gini Importance)", fontsize=11, fontweight='bold')
            ax_fi.set_xlabel("Tingkat Kepentingan Relatif")
            ax_fi.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_fi)
            
            st.markdown("""
            **Temuan Utama**:
            1. **`cp` (Jenis Nyeri Dada)** dan **`thalach` (Detak Jantung Maksimal)** merupakan prediktor dominan dalam mendeteksi iskemia miokard.
            2. **`oldpeak` (Depresi ST EKG)** dan **`ca` (Jumlah Pembuluh Darah Mayor Tersumbat)** memberikan indikasi anatomis kuat terhadap penyakit arteri koroner.
            """)

    with tab3:
        st.subheader("Eksplorasi Pola Medis Pasien (EDA)")
        st.markdown("Pola distribusi fitur klinis pada dataset Heart Disease:")

        try:
            df_raw = pd.read_csv("heart.csv").drop_duplicates()
            
            eda_col1, eda_col2 = st.columns(2)
            with eda_col1:
                st.markdown("**Distribusi Usia vs Detak Jantung Maksimal (`thalach`)**")
                fig_eda1, ax_eda1 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df_raw, x="age", y="thalach", hue="target",
                                palette={0: "#2a9d8f", 1: "#e63946"}, alpha=0.8, ax=ax_eda1)
                ax_eda1.set_xlabel("Usia (Tahun)")
                ax_eda1.set_ylabel("Detak Jantung Maksimum (bpm)")
                ax_eda1.legend(labels=["Sakit Jantung (1)", "Normal (0)"], title="Status")
                ax_eda1.grid(True, linestyle=':', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_eda1)
                st.caption("Pasien dengan penyakit jantung cenderung memiliki detak jantung maksimal yang lebih tinggi pada kelompok usianya.")

            with eda_col2:
                st.markdown("**Proporsi Penyakit Berdasarkan Nyeri Dada Saat Olahraga (`exang`)**")
                fig_eda2, ax_eda2 = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_raw, x="exang", hue="target",
                              palette={0: "#2a9d8f", 1: "#e63946"}, ax=ax_eda2)
                ax_eda2.set_xticklabels(["Tidak Ada Angina (0)", "Ada Angina Olahraga (1)"])
                ax_eda2.set_xlabel("Exercise Induced Angina")
                ax_eda2.set_ylabel("Jumlah Pasien")
                ax_eda2.legend(labels=["Normal (0)", "Sakit Jantung (1)"], title="Status")
                ax_eda2.grid(axis='y', linestyle=':', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_eda2)
                st.caption("Adanya nyeri dada yang dipicu aktivitas fisik berkolerasi signifikan dengan abnormalitas koroner.")
        except Exception as e:
            st.warning(f"Visualisasi EDA tidak dapat dimuat: {e}")


def page_detail_proyek():
    st.title("📚 Metodologi & Arsitektur Sistem")
    st.markdown("Dokumentasi teknis pipeline data science, dari akuisisi data hingga deployment.")
    st.divider()

    st.header("1. Ringkasan Arsitektur End-to-End")
    st.markdown("""
    ```mermaid
    graph TD;
        A[Raw Data: UCI Heart Disease 1025 baris] --> B[src/data: Deduplikasi 288 baris & Filter IQR];
        B --> C[src/features: StandardScaler & 5-Fold Stratified Split];
        C --> D[src/models: Multi-Model Training LR, DT, MLP, RF];
        D <--> E[(MLflow Tracking: Runs, Hyperparams & Metrics)];
        D --> F[Serialized Artifacts: Model, Scaler, Evaluasi];
        F --> G[Streamlit Web Application];
        G --> H[SHAP Explainability Engine];
        G --> I[Google Gemini API + Clinical Fallback];
    ```
    """)
    st.divider()

    st.header("2. Pipeline Data & Seleksi Fitur")
    st.markdown("""
    - **Pembersihan Data**: Dataset UCI asli memiliki 1025 baris dengan banyak duplikasi. Pipeline membersihkan menjadi **288 sampel unik** dan membersihkan *extreme outliers* pada `trestbps` dan `chol` menggunakan metode IQR.
    - **Seleksi 9 Fitur Kunci**: `cp`, `thalach`, `slope`, `oldpeak`, `exang`, `ca`, `thal`, `sex`, `age`.
    - **Validasi Silang**: Menerapkan **5-Fold Stratified Cross-Validation** untuk memastikan kestabilan estimasi generalisasi model di setiap fold.
    """)
    st.divider()

    st.header("3. Analisis Biaya Klinis: False Negative vs False Positive")
    st.markdown("""
    Dalam evaluasi Machine Learning untuk kesehatan:
    - **False Negative (Tipe II Error)**: Pasien sakit diprediksi sehat. Ini merupakan risiko kritis karena keterlambatan penanganan dapat berakibat fatal. Oleh karena itu, metrik **Recall / Sensitivitas** sangat diprioritaskan.
    - **False Positive (Tipe I Error)**: Pasien sehat diprediksi sakit. Konsekuensinya adalah pemeriksaan medis lanjutan yang menimbulkan kecemasan dan biaya tambahan, namun tidak membahayakan nyawa.
    - **Model Terpilih**: Random Forest tuned mencapai keseimbangan optimal dengan **Recall ~0.82-0.85** dan **AUC-ROC ~0.90**.
    """)


def page_tentang_saya():
    st.title("👤 Profil Pengembang")
    st.markdown("""
    Halo! Saya **Arfi Adi Nugroho**, mahasiswa Data Science dari Politeknik Elektronika Negeri Surabaya (PENS).
    Saya berfokus pada pengembangan sistem Machine Learning yang *production-ready*, terukur (*scalable*), dan transparan (*explainable*).

    ### 📬 Kontak & Portofolio:
    - **LinkedIn**: [Arfi Adi Nugroho](https://www.linkedin.com/in/arfi-adi-nugroho-58b0a7263)
    - **GitHub**: [github.com/Arfiadi](https://github.com/Arfiadi)
    - **Email**: arfiadi13@gmail.com
    """)


# --- Navigasi Sidebar ---
st.sidebar.title("🫀 Navigasi")
page_options = {
    "Home": page_home,
    "Form Prediksi Pasien": page_prediksi,
    "Eksplorasi Data (EDA) & Evaluasi": page_eda_dan_evaluasi,
    "Metodologi & Arsitektur": page_detail_proyek,
    "Tentang Pengembang": page_tentang_saya
}
selected_page = st.sidebar.radio("Pilih Halaman:", list(page_options.keys()))

page_options[selected_page]()