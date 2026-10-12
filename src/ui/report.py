"""
Clinical report generation utilities for patient screening summaries.
"""
from typing import Dict, Any


def generate_clinical_report(
    pred_res: Dict[str, Any],
    prediction: int,
    probability: float,
    rec_text: str
) -> str:
    """
    Constructs a formalized medical screening summary report formatted as text.
    Suitable for physician review or digital archive export.
    """
    timestamp = pred_res.get('timestamp', 'N/A')
    summary = pred_res.get('patient_summary', {})
    diagnosis = "Risiko Tinggi Penyakit Jantung" if prediction == 1 else "Risiko Rendah / Normal"
    confidence = max(probability, 1 - probability) * 100

    report = f"""====================================================
RINGKASAN SKRINING RISIKO PENYAKIT JANTUNG (CDSS)
====================================================
Waktu Skrining       : {timestamp}
Hasil Diagnosis      : {diagnosis}
Estimasi Probabilitas: {probability * 100:.2f}%
Model Keyakinan      : {confidence:.2f}% (Random Forest Tuned)

DATA REKAM MEDIS PASIEN:
----------------------------------------------------
- Usia               : {summary.get('Usia', '-')} tahun
- Jenis Kelamin      : {summary.get('Gender', '-')}
- Karakteristik Nyeri: {summary.get('Nyeri Dada', '-')}
- Detak Jantung Max  : {summary.get('Detak Max', '-')}
- Depresi Segmen ST  : {summary.get('Depresi ST', '-')}
- Angina Olahraga    : {summary.get('Angina Olahraga', '-')}
- Pembuluh Tersumbat : {summary.get('Pembuluh Tersumbat', '-')}
- Thallium Scan      : {summary.get('Thallium Scan', '-')}

SARAN MEDIS & GAYA HIDUP (AI CLINICAL ADVISOR):
----------------------------------------------------
{rec_text}

DISCLAIMER MEDIS:
----------------------------------------------------
Laporan ini dihasilkan secara otomatis oleh sistem Clinical Decision Support (CDSS)
berdasarkan pemodelan machine learning. Dokumen ini ditujukan semata-mata untuk skrining awal,
triase klinis, dan edukasi pasien. Hasil ini BUKAN pengganti diagnosis medis definitif oleh
dokter spesialis jantung dan pembuluh darah (Sp.JP).
====================================================
"""
    return report
