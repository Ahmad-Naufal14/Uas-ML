import os
import joblib
from sklearn.preprocessing import StandardScaler
from ml_core import load_and_preprocess_data, create_model

def run_training():
    data_path = 'data/winequality-white.csv'
    if not os.path.exists(data_path):
        data_path = '../data/winequality-white.csv'
        
    print("⏳ Memuat data dan memproses fitur...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    print("⚖️ Melatih komponen StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("🤖 Melatih Model Support Vector Machine (SVM)...")
    model = create_model()
    model.fit(X_train_scaled, y_train)
    
    # Simpan Artifact
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_wine_quality_model.joblib')
    joblib.dump(scaler, 'models/pipeline_scaler.joblib')
    print("✨ Model dan Scaler berhasil disimpan di folder 'models/'!")

if __name__ == "__main__":
    run_training()