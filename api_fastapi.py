from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Inisialisasi FastAPI
app = FastAPI(
    title="🍷 White Wine Quality Classifier API",
    description="API berbasis FastAPI untuk memprediksi kualitas White Wine menggunakan model SVM Optimized.",
    version="1.0.0"
)

# 2. Memuat Artifact Model & Scaler
MODEL_PATH = "models/best_wine_quality_model.joblib"
SCALER_PATH = "models/pipeline_scaler.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("❌ File model atau scaler tidak ditemukan di folder 'models/'. Jalankan src/train.py terlebih dahulu.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 3. Skema Data Input Menggunakan Pydantic (Sesuai 11 Fitur Kimia)
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# 4. Endpoint Root (Cek Status API)
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Welcome to White Wine Quality Prediction API. Go to /docs for Swagger UI."
    }

# 5. Endpoint Prediksi (POST Method)
@app.post("/predict")
def predict_wine_quality(data: WineFeatures):
    try:
        # Menyusun 12 kolom (11 fitur input + 1 dummy target kolom ke-12 agar cocok dengan StandardScaler)
        input_data = np.array([[
            data.fixed_acidity,
            data.volatile_acidity,
            data.citric_acid,
            data.residual_sugar,
            data.chlorides,
            data.free_sulfur_dioxide,
            data.total_sulfur_dioxide,
            data.density,
            data.pH,
            data.sulphates,
            data.alcohol,
            0.0  # Dummy target
        ]])
        
        # Transformasi menggunakan Scaler
        scaled_data = scaler.transform(input_data)
        
        # Eksekusi Prediksi & Probabilitas
        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]
        
        # Menentukan Label Output
        label = "PREMIUM QUALITY WINE" if prediction == 1 else "NORMAL QUALITY WINE"
        confidence = float(probabilities[1] if prediction == 1 else probabilities[0])
        
        return {
            "prediction": prediction,
            "label": label,
            "confidence_score": round(confidence * 100, 2),
            "status": "Success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan komputasi: {str(e)}")