import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_and_preprocess_data(filepath):
    # Membaca dataset
    df = pd.read_csv(filepath, sep=';') if ';' in open(filepath).read() else pd.read_csv(filepath)
    
    # Transformasi Target Biner (Premium >= 6, Normal < 6)
    df['target'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
    
    # Fitur (X) menggunakan 12 kolom agar konsisten dengan pipeline awal
    X = df.drop(columns=['quality'])
    y = df['target']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def create_model():
    # Model SVM Optimized dengan probabilitas diaktifkan
    return SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)