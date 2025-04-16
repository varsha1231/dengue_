import streamlit as st
import torch
import torch.nn as nn

class DengueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.net(x)

model = DengueNet()
model.load_state_dict(torch.load("dengue_model.pt", map_location=torch.device("cpu")))
model.eval()

def check_breeding_conditions(temp, humidity, rainfall):
    if 25 <= temp <= 30 and humidity >= 70 and rainfall >= 5:
        return "✅ Ideal for mosquito breeding"
    elif 25 <= temp <= 30 and (humidity >= 70 or rainfall >= 5):
        return "⚠️ Partially favorable"
    else:
        return "❄️ Not favorable"

st.title("🦟 Dengue Risk Predictor")
st.write("Enter today's weather data to check breeding conditions and dengue risk.")

temp = st.slider("🌡️ Temperature (°C)", 20, 40, 30)
humidity = st.slider("💧 Humidity (%)", 40, 100, 75)
rainfall = st.slider("🌧️ Rainfall (mm)", 0, 50, 10)

if st.button("Predict"):
    st.info(check_breeding_conditions(temp, humidity, rainfall))
    x = torch.tensor([[temp, humidity, rainfall]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
        risk = torch.argmax(y).item()
    levels = ["Low", "Medium", "High"]
    st.success(f"🚨 Predicted Dengue Risk: **{levels[risk]}**")
