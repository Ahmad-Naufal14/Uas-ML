import pandas as pd
import numpy as np

def generate_synthetic_wine_data(n_samples=5):
    np.random.seed(42)
    synthetic_data = {
        'fixed_acidity': np.random.uniform(5.0, 9.0, n_samples),
        'volatile_acidity': np.random.uniform(0.1, 0.5, n_samples),
        'citric_acid': np.random.uniform(0.2, 0.6, n_samples),
        'residual_sugar': np.random.uniform(1.0, 15.0, n_samples),
        'chlorides': np.random.uniform(0.02, 0.07, n_samples),
        'free_sulfur_dioxide': np.random.uniform(10.0, 60.0, n_samples),
        'total_sulfur_dioxide': np.random.uniform(60.0, 200.0, n_samples),
        'density': np.random.uniform(0.990, 0.999, n_samples),
        'pH': np.random.uniform(2.9, 3.5, n_samples),
        'sulphates': np.random.uniform(0.3, 0.7, n_samples),
        'alcohol': np.random.uniform(8.5, 13.0, n_samples),
        'target_dummy': np.zeros(n_samples) # Kolom ke-12 agar pas
    }
    df = pd.DataFrame(synthetic_data)
    df.to_csv('data/synthetic_wine_test.csv', index=False)
    print("💾 5 Sampel Data Lab Buatan sukses disimpan di 'data/synthetic_wine_test.csv'!")

if __name__ == "__main__":
    generate_synthetic_wine_data()