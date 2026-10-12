import os
import sys
import time
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is on sys.path for relative submodule resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.services.llm_advisor import get_lifestyle_recommendation
from src.ui import (
    apply_custom_styles,
    render_system_banner,
    plot_risk_gauge,
    plot_shap_summary,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_eda_scatter,
    plot_eda_bar,
    generate_clinical_report,
)

# --- Base Directory & File Paths (Streamlit Cloud Safe) ---
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'heart_disease_artifacts.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'heart.csv')
IMAGE_PATH = (
    os.path.join(BASE_DIR, 'assets', 'heart.jpg')
    if os.path.exists(os.path.join(BASE_DIR, 'assets', 'heart.jpg'))
    else os.path.join(BASE_DIR, 'heart.jpg')
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Clinical Decision Support System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply Medical Design System ---
apply_custom_styles()


# --- Artifact Loading ---
@st.cache_resource
def load_artifacts():
    """
    Load serialized model, scaler, and evaluation metrics safely with fallback paths.
    """
    path_candidates = [
        ARTIFACTS_PATH,
        'heart_disease_artifacts.pkl',
        os.path.join(os.getcwd(), 'heart_disease_artifacts.pkl')
    ]
    for p in path_candidates:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as file:
                    return pickle.load(file)
            except Exception as e:
                st.error(f"Gagal memuat artefak model dari {p}: {e}")
                return None
    return None


artifacts = load_artifacts()
model = artifacts['best_model'] if artifacts else None
scaler = artifacts['scaler'] if artifacts else None
feature_names = artifacts.get('feature_names', [
    'cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age'
]) if artifacts else [
    'cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age'
]


# --- Helper Prediction Functions ---
def predict_heart_disease(input_data):
    df_input = pd.DataFrame([input_data], columns=feature_names)
    scaled_data = scaler.transform(df_input)
    prediction = model.predict(scaled_data)
    return int(prediction[0])


def predict_heart_disease_proba(input_data):
    df_input = pd.DataFrame([input_data], columns=feature_names)
    scaled_data = scaler.transform(df_input)
    return float(model.predict_proba(scaled_data)[0, 1])


@st.cache_data(show_spinner=False)
def cached_lifestyle_recommendation(age: int, sex: int, thalach: int, cp: str, prediction: int) -> str:
    patient_data_dict = {
        'age': age,
        'sex': sex,
        'thalach': thalach,
        'cp': cp
    }
    return get_lifestyle_recommendation(patient_data_dict, prediction)


# --- PAGE 1: RINGKASAN EKSEKUTIF & BERANDA ---
def page_home():
    render_system_banner()

    st.title("🫀 Clinical Decision Support System: Skrining Penyakit Jantung")
    st.markdown("""
    Sistem berbasis kecerdasan buatan (*AI Clinical Decision Support System*) yang mengintegrasikan 
    **Machine Learning Terkalibrasi (Random Forest Tuned, AUC 0.898)**, **Explainable AI (SHAP)**, dan **Medical AI Advisor** 
    guna membantu klinisi serta pasien mendapatkan estimasi risiko dini yang transparan dan dapat dipertanggungjawabkan.
    """)
    st.divider()

    # Top KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(label="Model ROC-AUC", value="0.898", delta="+0.04 vs Baseline")
    with col_kpi2:
        st.metric(label="5-Fold CV Accuracy", value="83.0%", delta="±5.4% Stability")
    with col_kpi3:
        st.metric(label="Clinical Recall", value="82.4%", delta="Prioritas Sensitivitas")
    with col_kpi4:
        st.metric(label="Explainability Engine", value="SHAP Tree", delta="Game Theory XAI")

    st.markdown("---")

    col_info, col_img = st.columns([1.2, 1])
    with col_info:
        st.subheader("💡 Mengapa Pendekatan Sistem Ini Berbeda?")
        st.markdown("""
        1. **Rigor Validasi Silang (Anti Lucky-Split)**:
           Model dievaluasi menggunakan *5-Fold Stratified Cross-Validation* dan *20% holdout test set* independen untuk menjamin kestabilan prediksi pada data baru.
        2. **Optimasi Biaya Klinis (False Negative vs False Positive)**:
           Dalam domain kardiologi, **False Negative** (pasien sakit dikira normal) berisiko fatal. Model dirancang memprioritaskan sensitivitas tinggi tanpa membebani fasilitas kesehatan dengan *over-diagnosis*.
        3. **Explainable AI Transparan**:
           Bukan model *black-box*. Setiap keputusan prediksi dibedah kontribusi fiturnya secara interaktif menggunakan nilai SHAP.
        4. **Rekomendasi Gaya Hidup Tahan Banting**:
           Terintegrasi dengan Google Gemini AI dan dilengkapi protokol medis berbasis aturan (*graceful fallback*) yang aktif otomatis saat koneksi offline.
        """)
        st.info("👉 **Mulai Skrining**: Gunakan menu **'Form Skrining Pasien'** di sidebar untuk mencoba analisis dengan kasus contoh 1-klik.")

    with col_img:
        image_loaded = False
        if os.path.exists(IMAGE_PATH):
            try:
                img = Image.open(IMAGE_PATH)
                st.image(img, caption="Cardiovascular Clinical Care Platform", use_container_width=True)
                image_loaded = True
            except Exception:
                pass
        if not image_loaded:
            st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=800", caption="Cardiovascular Care", use_container_width=True)


# --- PAGE 2: FORM PREDIKSI PASIEN ---
def page_prediksi():
    render_system_banner()

    st.title("🩺 Form Skrining Klinis & Stratifikasi Risiko")
    st.markdown("Isi rekam medis pasien di bawah ini atau gunakan **Preset Kasus Cepat** untuk mengisi profil pasien contoh secara instan.")

    if model is None or scaler is None:
        st.error("Artefak model tidak ditemukan. Pastikan file `heart_disease_artifacts.pkl` tersedia.")
        return

    # Inisialisasi Session State untuk Profil Pasien
    if 'profile_input' not in st.session_state:
        st.session_state.profile_input = {
            'age': 54, 'sex': 'Pria', 'cp': 'Typical Angina (Nyeri Dada Tipikal)',
            'thalach': 150, 'oldpeak': 1.0, 'slope': 'Upsloping (Menaik)',
            'exang': 'Tidak', 'ca': '0', 'thal': 'Normal'
        }

    # --- QUICK PRESET BUTTONS (1-CLICK RECRUITER DEMO) ---
    st.subheader("⚡ Quick Case Presets (Contoh Kasus 1-Klik)")
    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        if st.button("🚨 Kasus A: Risiko Tinggi", use_container_width=True, help="Pasien pria lansia dengan gejala angina tipikal dan depresi ST tinggi"):
            st.session_state.profile_input = {
                'age': 63, 'sex': 'Pria', 'cp': 'Typical Angina (Nyeri Dada Tipikal)',
                'thalach': 125, 'oldpeak': 2.4, 'slope': 'Flat (Datar)',
                'exang': 'Ya', 'ca': '2', 'thal': 'Reversible Defect (Cacat Reversibel)'
            }
            st.toast("Preset Kasus Risiko Tinggi Terpilih!", icon="🚨")

    with p_col2:
        if st.button("💚 Kasus B: Risiko Rendah", use_container_width=True, help="Pasien wanita usia produktif dengan detak jantung prima dan EKG normal"):
            st.session_state.profile_input = {
                'age': 41, 'sex': 'Wanita', 'cp': 'Non-anginal Pain (Bukan Angina)',
                'thalach': 176, 'oldpeak': 0.2, 'slope': 'Upsloping (Menaik)',
                'exang': 'Tidak', 'ca': '0', 'thal': 'Normal'
            }
            st.toast("Preset Kasus Risiko Rendah Terpilih!", icon="💚")

    with p_col3:
        if st.button("⚠️ Kasus C: Kasus Borderline", use_container_width=True, help="Pasien paruh baya dengan parameter campuran dan riwayat defect tetap"):
            st.session_state.profile_input = {
                'age': 58, 'sex': 'Pria', 'cp': 'Atypical Angina (Nyeri Dada Atipikal)',
                'thalach': 142, 'oldpeak': 1.2, 'slope': 'Flat (Datar)',
                'exang': 'Tidak', 'ca': '1', 'thal': 'Fixed Defect (Cacat Permanen)'
            }
            st.toast("Preset Kasus Borderline Terpilih!", icon="⚠️")

    st.markdown("---")

    # Options mapping dictionaries
    sex_options = {'Pria': 1, 'Wanita': 0}
    cp_options = {
        'Typical Angina (Nyeri Dada Tipikal)': 0,
        'Atypical Angina (Nyeri Dada Atipikal)': 1,
        'Non-anginal Pain (Bukan Angina)': 2,
        'Asymptomatic (Tanpa Gejala Nyeri)': 3
    }
    slope_options = {'Upsloping (Menaik)': 2, 'Flat (Datar)': 1, 'Downsloping (Menurun)': 0}
    exang_options = {'Ya': 1, 'Tidak': 0}
    ca_options = {'0': 0, '1': 1, '2': 2, '3': 3}
    thal_options = {'Normal': 1, 'Fixed Defect (Cacat Permanen)': 2, 'Reversible Defect (Cacat Reversibel)': 3}

    # --- STRUCTURED MEDICAL INPUT FORM ---
    with st.form("clinical_intake_form"):
        col_sec1, col_sec2, col_sec3 = st.columns(3)

        with col_sec1:
            st.markdown("#### 👤 1. Profil Demografi")
            current_age = st.session_state.profile_input['age']
            age = st.slider('Usia Pasien (age)', 29, 77, current_age, help="Usia pasien dalam tahun.")

            current_sex = st.session_state.profile_input['sex']
            sex_idx = list(sex_options.keys()).index(current_sex) if current_sex in sex_options else 0
            sex = st.radio('Jenis Kelamin (sex)', list(sex_options.keys()), index=sex_idx, horizontal=True)

        with col_sec2:
            st.markdown("#### 💓 2. Gejala & Tanda Vital")
            current_cp = st.session_state.profile_input['cp']
            cp_idx = list(cp_options.keys()).index(current_cp) if current_cp in cp_options else 0
            cp = st.selectbox('Karakteristik Nyeri Dada (cp)', list(cp_options.keys()), index=cp_idx, help="Jenis nyeri dada klinis.")

            current_thalach = st.session_state.profile_input['thalach']
            thalach = st.slider('Detak Jantung Maksimal (thalach)', 71, 202, current_thalach, help="Detak puncak saat tes latihan (bpm).")

            current_exang = st.session_state.profile_input['exang']
            exang_idx = list(exang_options.keys()).index(current_exang) if current_exang in exang_options else 1
            exang = st.selectbox('Nyeri Dada Akibat Olahraga (exang)', list(exang_options.keys()), index=exang_idx, help="Apakah angina timbul saat latihan fisik?")

        with col_sec3:
            st.markdown("#### 📈 3. Uji EKG & Diagnostik")
            current_oldpeak = float(st.session_state.profile_input['oldpeak'])
            oldpeak = st.slider('Depresi Segmen ST EKG (oldpeak)', 0.0, 6.2, current_oldpeak, step=0.1, help="Penurunan ST segmen saat latihan dibanding istirahat.")

            current_slope = st.session_state.profile_input['slope']
            slope_idx = list(slope_options.keys()).index(current_slope) if current_slope in slope_options else 0
            slope = st.selectbox('Kemiringan Segmen ST (slope)', list(slope_options.keys()), index=slope_idx, help="Kemiringan segmen ST pada EKG.")

            current_ca = st.session_state.profile_input['ca']
            ca_idx = list(ca_options.keys()).index(current_ca) if current_ca in ca_options else 0
            ca = st.selectbox('Pembuluh Darah Mayor Tersumbat (ca)', list(ca_options.keys()), index=ca_idx, help="Jumlah pembuluh darah utama (0-3) terfluoroskopi.")

            current_thal = st.session_state.profile_input['thal']
            thal_idx = list(thal_options.keys()).index(current_thal) if current_thal in thal_options else 0
            thal = st.selectbox('Hasil Thallium Scan (thal)', list(thal_options.keys()), index=thal_idx, help="Status perfusi miokard Thallium.")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Jalankan Analisis & Skrining Kardiovaskular", type="primary", use_container_width=True)

    # --- HASIL DIAGNOSIS JIKA SUBMITTED ---
    if submitted:
        input_data = [
            cp_options[cp], thalach, slope_options[slope], oldpeak,
            exang_options[exang], ca_options[ca], thal_options[thal],
            sex_options[sex], age
        ]

        with st.spinner('Mengevaluasi parameter klinis dengan Random Forest model...'):
            time.sleep(0.3)
            prediction = predict_heart_disease(input_data)
            probability = predict_heart_disease_proba(input_data)

        # Simpan hasil terakhir di session state
        st.session_state.last_prediction = {
            'prediction': prediction,
            'probability': probability,
            'input_data': input_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'patient_summary': {
                'Usia': age,
                'Gender': sex,
                'Nyeri Dada': cp,
                'Detak Max': f"{thalach} bpm",
                'Depresi ST': f"{oldpeak} mm",
                'Angina Olahraga': exang,
                'Pembuluh Tersumbat': ca,
                'Thallium Scan': thal
            }
        }

    # Render Hasil jika sudah ada prediksi
    if 'last_prediction' in st.session_state:
        pred_res = st.session_state.last_prediction
        prediction = pred_res['prediction']
        probability = pred_res['probability']
        input_data = pred_res['input_data']

        st.markdown("---")
        st.subheader("📋 Ringkasan Hasil Diagnosis & Stratifikasi Risiko")

        col_gauge, col_verdict = st.columns([1.2, 1])

        with col_gauge:
            fig_gauge = plot_risk_gauge(probability)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_verdict:
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("""
                <div style="border-left: 5px solid #DC2626; background-color: #FEF2F2; padding: 18px; border-radius: 8px;">
                    <h3 style="color: #991B1B; margin: 0 0 8px 0;">💔 Indikasi Risiko Tinggi Kardiovaskular</h3>
                    <p style="color: #7F1D1D; margin: 0; font-size: 0.95rem;">
                        Model mengidentifikasi pola biomarker klinis yang mengindikasikan adanya iskemia atau abnormalitas perfusi arteri koroner. 
                        Disarankan untuk melakukan pemeriksaan lanjutan (Treadmill Test / Angiografi).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="border-left: 5px solid #059669; background-color: #ECFDF5; padding: 18px; border-radius: 8px;">
                    <h3 style="color: #065F46; margin: 0 0 8px 0;">❤️ Estimasi Risiko Rendah / Normal</h3>
                    <p style="color: #064E3B; margin: 0; font-size: 0.95rem;">
                        Parameter klinis pasien berada dalam rentang toleransi fisiologis normal. Pertahankan pola hidup sehat dan kontrol rutin berkala.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**Keyakinan Model:** `{max(probability, 1 - probability) * 100:.1f}%` • **Algoritma:** `Random Forest (Tuned)`")

        # --- EXPLAINABLE AI (SHAP) PLOTLY CHART ---
        st.markdown("---")
        st.subheader("🔍 Explainable AI (XAI): Kontribusi Faktor Risiko Pasien (SHAP)")
        st.write("Grafik interaktif ini menguraikan seberapa besar setiap parameter medis mendorong (**merah / meningkatkan risiko**) atau melindungi (**hijau / menurunkan risiko**) hasil diagnosis pasien ini.")

        try:
            fig_shap = plot_shap_summary(model, scaler, input_data, feature_names)
            if fig_shap:
                st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"Visualisasi SHAP tidak dapat dimuat: {e}")

        # --- AI ADVISOR & CLINICAL GUIDELINE ---
        st.markdown("---")
        st.subheader("💡 Rekomendasi Medis & Konsultasi Gaya Hidup")

        with st.spinner("Menghubungkan ke Clinical AI Advisor..."):
            rec_text = cached_lifestyle_recommendation(
                pred_res['patient_summary']['Usia'],
                sex_options[pred_res['patient_summary']['Gender']],
                int(pred_res['patient_summary']['Detak Max'].replace(' bpm', '')),
                pred_res['patient_summary']['Nyeri Dada'],
                prediction
            )
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 22px;">
                {rec_text}
            </div>
            """, unsafe_allow_html=True)

        # --- EXPORT REPORT BUTTON ---
        st.markdown("<br>", unsafe_allow_html=True)
        report_content = generate_clinical_report(pred_res, prediction, probability, rec_text)
        st.download_button(
            label="📄 Unduh Ringkasan Laporan Klinis Pasien (.txt)",
            data=report_content,
            file_name=f"Laporan_Skrining_Jantung_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


# --- PAGE 3: EKSPLORASI DATA & EVALUASI MODEL ---
def page_eda_dan_evaluasi():
    render_system_banner()

    st.title("📊 Eksplorasi Data (EDA) & Evaluasi Multi-Model")
    st.markdown("Transparansi metrik evaluasi model, perbandingan 4 keluarga algoritma di MLflow, dan eksplorasi data klinis interaktif.")
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "🏆 Evaluasi & Perbandingan Model (MLflow)",
        "📈 Feature Importance (MDI vs SHAP)",
        "🔬 Eksplorasi Pola Medis (Interactive EDA)"
    ])

    with tab1:
        st.subheader("1. Perbandingan Model Kandidat (Stratified 5-Fold CV & Test Set)")
        st.markdown("""
        Evaluasi mencakup 4 keluarga algoritma yang dilacak secara otomatis di **MLflow**:
        *Linear Model* (Logistic Regression), *Tree Model* (Decision Tree), *Neural Network* (MLP), dan *Ensemble Learning* (Random Forest Tuned).
        """)

        if artifacts and "model_comparison" in artifacts:
            df_comp = artifacts["model_comparison"]
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            fig_comp = plot_model_comparison(df_comp)
            st.plotly_chart(fig_comp, use_container_width=True)

        st.divider()

        st.subheader("2. Evaluasi Diagnostik: Confusion Matrix & Kurva ROC Interaktif")
        col_cm, col_roc = st.columns(2)

        if artifacts and "test_eval" in artifacts:
            eval_data = artifacts["test_eval"]
            cm = eval_data["confusion_matrix"]

            with col_cm:
                st.markdown("**Confusion Matrix (Holdout Test Set)**")
                fig_cm = plot_confusion_matrix(cm)
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption("📌 **Prioritas Klinis**: Menekan False Negative (pasien sakit lolos) adalah target primer untuk mencegah serangan jantung tanpa penanganan.")

            with col_roc:
                roc_val = eval_data.get('roc_auc', 0.898)
                st.markdown(f"**Receiver Operating Characteristic (ROC Curve) - AUC: {roc_val:.3f}**")
                fig_roc = plot_roc_curve(eval_data["fpr"], eval_data["tpr"], roc_val)
                st.plotly_chart(fig_roc, use_container_width=True)
                st.caption("Nilai AUC mendekati 0.90 menunjukkan kemampuan diskriminasi kelas yang luar biasa.")

    with tab2:
        st.subheader("Analisis Pentingnya Fitur (Random Forest MDI Gini vs SHAP)")
        st.markdown("""
        Dua perspektif interpretasi fitur:
        - **Gini Importance (MDI)**: Menghitung seberapa efektif pemisahan node pohon keputusan di seluruh ensemble Random Forest.
        - **SHAP Summary**: Menghitung kontribusi marjinal berbasis teori permainan (*cooperative game theory*).
        """)

        if artifacts and "feature_importance" in artifacts:
            df_feat = artifacts["feature_importance"]
            fig_fi = plot_feature_importance(df_feat)
            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown("""
            **Temuan Utama Interpretasi Klinis**:
            1. **`cp` (Nyeri Dada)** & **`thalach` (Detak Jantung Puncak)**: Dua indikator fungsional paling dominan dalam membedakan ada/tidaknya iskemia koroner.
            2. **`oldpeak` & `ca`**: Dua parameter penanda fisik anatomis yang mengonfirmasi adanya kerusakan segmen EKG dan penyumbatan pembuluh darah mayor.
            """)

    with tab3:
        st.subheader("Eksplorasi Pola Medis Interaktif (EDA)")
        try:
            csv_path = DATA_PATH if os.path.exists(DATA_PATH) else "heart.csv"
            df_raw = pd.read_csv(csv_path).drop_duplicates()
            df_raw['Status'] = df_raw['target'].map({0: 'Normal / Sehat (0)', 1: 'Penyakit Jantung (1)'})

            eda_col1, eda_col2 = st.columns(2)

            with eda_col1:
                st.markdown("**Distribusi Usia vs Detak Jantung Maksimal (`thalach`)**")
                fig_scatter = plot_eda_scatter(df_raw)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("Kelompok usia 40-60 dengan detak jantung maksimal tinggi cenderung memiliki prevalensi terdiagnosis lebih tinggi saat tes stres.")

            with eda_col2:
                st.markdown("**Proporsi Penyakit Berdasarkan Nyeri Dada Saat Olahraga (`exang`)**")
                fig_bar = plot_eda_bar(df_raw)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption("Adanya angina yang dipicu aktivitas fisik berkolerasi signifikan dengan abnormalitas koroner.")
        except Exception as e:
            st.warning(f"Visualisasi EDA tidak dapat dimuat: {e}")


# --- PAGE 4: METODOLOGI & ARSITEKTUR ---
def page_detail_proyek():
    render_system_banner()

    st.title("📚 Metodologi & Arsitektur Sistem")
    st.markdown("Dokumentasi teknis pipeline data science, pemodelan, dan pertimbangan keselamatan medis.")
    st.divider()

    st.header("1. Arsitektur End-to-End Pipeline")
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
        G --> J[Plotly Interactive Visualizations];
    ```
    """)
    st.divider()

    st.header("2. Analisis Biaya Klinis: False Negative vs False Positive")
    st.markdown("""
    Dalam evaluasi Machine Learning untuk pelayanan kesehatan:
    - **False Negative (Tipe II Error)**: Pasien sakit diprediksi sehat. Ini merupakan risiko paling fatal karena pasien tidak mendapatkan penanganan dini yang dapat berujung pada infark miokard akut. Oleh karena itu, metrik **Recall / Sensitivitas** sangat diprioritaskan.
    - **False Positive (Tipe I Error)**: Pasien sehat diprediksi sakit. Konsekuensinya adalah pemeriksaan medis lanjutan yang menimbulkan biaya dan kecemasan, namun tidak membahayakan nyawa.
    - **Model Terpilih**: Random Forest tuned mencapai keseimbangan optimal dengan **Recall 82.4%**, **Precision 87.5%**, dan **AUC-ROC 0.898**.
    """)


# --- PAGE 5: TENTANG PENGEMBANG ---
def page_tentang_saya():
    render_system_banner()

    st.title("👤 Profil Pengembang")
    st.markdown("""
    Halo! Saya **Arfi Adi Nugroho**, mahasiswa Data Science dari Politeknik Elektronika Negeri Surabaya (PENS).
    Saya berfokus pada pengembangan sistem Machine Learning yang *production-ready*, terukur (*scalable*), transparan (*explainable*), dan berorientasi pada dampak nyata.

    ### 📬 Kontak & Portofolio:
    - **LinkedIn**: [Arfi Adi Nugroho](https://www.linkedin.com/in/arfi-adi-nugroho-58b0a7263)
    - **GitHub**: [github.com/Arfiadi](https://github.com/Arfiadi)
    - **Email**: arfiadi13@gmail.com
    """)


# --- NAVIGASI SIDEBAR ---
st.sidebar.title("🫀 Navigasi")
page_options = {
    "Home / Overview": page_home,
    "Form Skrining Pasien": page_prediksi,
    "Eksplorasi Data & Evaluasi": page_eda_dan_evaluasi,
    "Metodologi & Arsitektur": page_detail_proyek,
    "Tentang Pengembang": page_tentang_saya
}
selected_page = st.sidebar.radio("Pilih Modul:", list(page_options.keys()))

page_options[selected_page]()