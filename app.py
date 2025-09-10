import gradio as gr
import pandas as pd
import joblib

# Load model + scaler
model = joblib.load("bodyfat_model_rf.pkl")
scaler = joblib.load("scaler_rf.pkl")

# Feature order
FEATURES = [
    "Age", "Weight", "Height", "Neck", "Chest", "Abdomen",
    "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"
]

def predict_bodyfat(age, weight, height, neck, chest, abdomen,
                    hip, thigh, knee, ankle, biceps, forearm, wrist):
    data = pd.DataFrame([[
        age, weight, height, neck, chest, abdomen,
        hip, thigh, knee, ankle, biceps, forearm, wrist
    ]], columns=FEATURES)

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    return f"Estimated Body Fat: {prediction:.2f}%"

inputs = [
    gr.Slider(10, 100, value=25, step=1, label="Age"),
    gr.Slider(30, 200, value=70, step=0.5, label="Weight (kg)"),
    gr.Slider(120, 220, value=170, step=0.5, label="Height (cm)"),
    gr.Slider(20, 60, value=35, step=0.5, label="Neck (cm)"),
    gr.Slider(60, 150, value=95, step=0.5, label="Chest (cm)"),
    gr.Slider(60, 150, value=85, step=0.5, label="Abdomen (cm)"),
    gr.Slider(70, 160, value=95, step=0.5, label="Hip (cm)"),
    gr.Slider(30, 90, value=55, step=0.5, label="Thigh (cm)"),
    gr.Slider(20, 60, value=38, step=0.5, label="Knee (cm)"),
    gr.Slider(15, 40, value=22, step=0.5, label="Ankle (cm)"),
    gr.Slider(20, 60, value=32, step=0.5, label="Biceps (cm)"),
    gr.Slider(15, 50, value=28, step=0.5, label="Forearm (cm)"),
    gr.Slider(10, 30, value=18, step=0.5, label="Wrist (cm)"),
]

output = gr.Textbox(label="Prediction")

app = gr.Interface(
    fn=predict_bodyfat,
    inputs=inputs,
    outputs=output,
    title="🏋️ Body Fat Estimator",
    description="Enter your body measurements using sliders to estimate your body fat percentage."
)

if __name__ == "__main__":
    app.launch()
