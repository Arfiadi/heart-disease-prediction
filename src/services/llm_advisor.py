import os
import requests

def load_env():
    """Manual fallback to load .env without strictly requiring python-dotenv"""
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

load_env()

def get_lifestyle_recommendation(patient_data: dict, prediction_result: int) -> str:
    """
    Calls the Google Gemini REST API directly using requests.
    Zero dependency conflicts, ultra-fast and reliable.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        return "⚠️ API Key Gemini belum dikonfigurasi. Silakan isi file `.env` dengan GEMINI_API_KEY Anda untuk mendapatkan saran gaya hidup otomatis berbasis AI."
        
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
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
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
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        if response.status_code == 200:
            result = response.json()
            candidates = result.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"]
            return "Tidak ada respons teks yang dihasilkan dari AI."
        else:
            return f"⚠️ Respon Gemini API (Status {response.status_code}): {response.text}"
    except Exception as e:
        return f"Gagal menghubungi Gemini AI: {str(e)}"
