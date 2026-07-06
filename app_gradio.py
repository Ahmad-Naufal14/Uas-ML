import gradio as gr
import joblib
import numpy as np

# Load model & scaler
model = joblib.load('models/best_wine_quality_model.joblib')
scaler = joblib.load('models/pipeline_scaler.joblib')

def predict_wine_gradio(fixed_acid, volatile_acid, citric, sugar, chlor, free_so2, total_so2, dens, ph_val, sulph, alc):
    # Menyusun 12 kolom dummy
    inputs = np.array([[fixed_acid, volatile_acid, citric, sugar, chlor, free_so2, total_so2, dens, ph_val, sulph, alc, 0]])
    scaled_inputs = scaler.transform(inputs)
    
    pred = model.predict(scaled_inputs)[0]
    proba = model.predict_proba(scaled_inputs)[0]
    
    status = "🎉 PREMIUM QUALITY WINE" if pred == 1 else "⚠️ NORMAL QUALITY WINE"
    kepastian = proba[1] if pred == 1 else proba[0]
    
    return f"Hasil Klasifikasi: {status}\nPersentase Kepastian Sistem: {kepastian * 100:.2f}%"

# Membuat Interface GUI Gradio
demo = gr.Interface(
    fn=predict_wine_gradio,
    inputs=[
        gr.Slider(3.0, 15.0, value=6.8, label="Fixed Acidity"),
        gr.Slider(0.0, 2.0, value=0.27, label="Volatile Acidity"),
        gr.Slider(0.0, 1.5, value=0.36, label="Citric Acid"),
        gr.Slider(0.0, 30.0, value=6.1, label="Residual Sugar"),
        gr.Slider(0.0, 0.5, value=0.042, label="Chlorides"),
        gr.Slider(1.0, 150.0, value=35.0, label="Free Sulfur Dioxide"),
        gr.Slider(5.0, 350.0, value=125.0, label="Total Sulfur Dioxide"),
        gr.Slider(0.98, 1.02, value=0.994, label="Density"),
        gr.Slider(2.5, 4.0, value=3.18, label="pH cairan"),
        gr.Slider(0.1, 2.0, value=0.48, label="Sulphates"),
        gr.Slider(7.0, 16.0, value=10.5, label="Alcohol (%)"),
    ],
    outputs="text",
    title="🍷 White Wine Quality Classifier - Gradio Interface",
    description="Interface alternatif untuk pengujian klasifikasi kualitas White Wine berbasis model SVM Optimized."
)

if __name__ == "__main__":
    demo.launch()