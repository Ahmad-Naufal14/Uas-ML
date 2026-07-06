import joblib
import numpy as np

def predict_single_sample(features_list):
    # Memuat artifact
    model = joblib.load('models/best_wine_quality_model.joblib')
    scaler = joblib.load('models/pipeline_scaler.joblib')
    
    # Transformasi input dengan dummy kolom ke-12
    features_array = np.array([features_list])
    features_scaled = scaler.transform(features_array)
    
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    return "PREMIUM QUALITY" if pred == 1 else "NORMAL QUALITY", prob

if __name__ == "__main__":
    # Contoh data dummy input 12 kolom (11 fitur kimia + 1 kolom dummy target)
    sample_data = [6.8, 0.27, 0.36, 6.1, 0.042, 35.0, 125.0, 0.9940, 3.18, 0.48, 10.5, 0]
    hasil, probabilitas = predict_single_sample(sample_data)
    print(f"Hasil Uji Coba Terminal: {hasil} (Probabilitas Kelas Premium: {probabilitas[1]*100:.2f}%)")