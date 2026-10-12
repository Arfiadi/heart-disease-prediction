import os
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

def ensure_env_loaded():
    """Lazily load .env without module-level side-effects."""
    if not os.getenv("GEMINI_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            if os.path.exists(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()

def get_rule_based_fallback(patient_data: Dict[str, Any], prediction_result: int) -> str:
    """
    High-quality clinical guideline-based lifestyle recommendations.
    Used as an automatic fallback when API key is missing or external service fails.
    """
    age = patient_data.get('age', 50)
    sex = "Pria" if patient_data.get('sex', 1) == 1 else "Wanita"
    
    if prediction_result == 1:
        return (
            "### Rekomendasi Gaya Hidup & Manajemen Risiko\n\n"
            f"*Catatan: Rekomendasi berbasis protokol klinis (Pasien {sex}, Usia {age} tahun - Kategori Risiko Tinggi).*\n\n"
            "1. **Evaluasi Kardiovaskular Lanjutan**: Segera jadwalkan konsultasi dengan dokter spesialis jantung "
            "untuk pemeriksaan diagnostik lanjutan (seperti Treadmill Stress Test atau Ekokardiografi).\n"
            "2. **Pola Makan Ramah Jantung**: Terapkan diet DASH/Mediterania dengan mengurangi natrium (< 2.000 mg/hari), "
            "membatasi lemak jenuh, dan meningkatkan asupan serat larut.\n"
            "3. **Aktivitas Fisik Terpantau**: Hindari olahraga berat yang mendadak. Mulailah latihan aerobik ringan "
            "seperti jalan santai 20-30 menit per hari setelah mendapatkan persetujuan dokter.\n\n"
            "---\n*Disclaimer: Saran ini disusun berdasarkan pedoman umum kesehatan jantung dan bukan pengganti nasihat medis profesional.*"
        )
    else:
        return (
            "### Rekomendasi Gaya Hidup Preventif\n\n"
            f"*Catatan: Rekomendasi berbasis protokol preventif (Pasien {sex}, Usia {age} tahun - Kategori Risiko Rendah).*\n\n"
            "1. **Pertahankan Kebugaran Fisik**: Lakukan aktivitas fisik intensitas sedang minimal 150 menit per minggu "
            "(seperti jalan cepat, bersepeda, atau berenang).\n"
            "2. **Nutrisi Seimbang & Kontrol Berat Badan**: Jaga berat badan ideal, konsumsi buah, sayur, kacang-kacangan, "
            "dan batasi konsumsi gula tambahan serta makanan olahan.\n"
            "3. **Pemeriksaan Kesehatan Rutin**: Lakukan medical check-up tahunan secara berkala untuk memantau tekanan darah, "
            "profil lipid (kolesterol), dan gula darah puasa.\n\n"
            "---\n*Disclaimer: Saran ini disusun berdasarkan pedoman umum kesehatan jantung dan bukan pengganti nasihat medis profesional.*"
        )

def get_lifestyle_recommendation(patient_data: Dict[str, Any], prediction_result: int) -> str:
    """
    Calls the Google Gemini REST API directly with graceful degradation to clinical rules.
    """
    ensure_env_loaded()
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Graceful fallback if API key is not configured
    if not api_key or api_key.strip() in ("", "your_google_gemini_api_key_here"):
        logger.info("GEMINI_API_KEY not configured. Using rule-based clinical fallback.")
        return get_rule_based_fallback(patient_data, prediction_result)
        
    status = "berisiko tinggi menderita penyakit jantung" if prediction_result == 1 else "memiliki risiko rendah menderita penyakit jantung"
    
    prompt = f"""
    Anda adalah seorang konsultan kesehatan AI profesional.
    Seorang pasien memiliki data medis berikut:
    - Usia: {patient_data.get('age', 'Tidak diketahui')} tahun
    - Jenis Kelamin: {'Pria' if patient_data.get('sex', 1) == 1 else 'Wanita'}
    - Detak Jantung Maksimal (thalach): {patient_data.get('thalach', 'Tidak diketahui')}
    - Nyeri Dada (cp): {patient_data.get('cp', 'Tidak diketahui')}
    
    Berdasarkan model AI kami, pasien ini {status}.
    
    Tolong berikan 3 poin saran gaya hidup singkat, suportif, dan relevan dengan kondisi mereka.
    Gunakan bahasa Indonesia yang sopan dan profesional.
    Tambahkan disclaimer singkat di akhir bahwa ini bukan nasihat medis pengganti dokter.
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                return candidates[0]["content"]["parts"][0]["text"]
        logger.warning("Gemini API call returned status %s: %s. Falling back to rule-based recommendation.", response.status_code, response.text)
    except Exception as e:
        logger.warning("Error calling Gemini API: %s. Falling back to rule-based recommendation.", e)
        
    return get_rule_based_fallback(patient_data, prediction_result)
