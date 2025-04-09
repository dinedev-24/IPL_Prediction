import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Set Streamlit configuration
st.set_page_config(page_title="🏏 Match Scenario Simulator")
st.title("🏏 Match Scenario Simulator")

# Load model and scalers
try:
    gru_model = load_model("gru_score_predictor.keras", compile=False)
    scaler_input = joblib.load("scaler_input.save")
    scaler_output = joblib.load("scaler_output.save")
except Exception as e:
    st.error(f"Model or scaler files not found: {e}")
    st.stop()

# User Input Section
st.subheader("📝 Input Current Match Situation")
st.markdown("🔢 Enter **runs scored per over**, not cumulative. For example:\n- Over 1: 6\n- Over 2: 9\n- Over 3: 0")

# Number of completed overs
current_over = st.slider("Select Over Completed (1 to 20)", 1, 20, 10)

# Input per-over runs
runs_input = []
for i in range(current_over):
    run = st.number_input(f"Runs in Over {i+1}", min_value=0, max_value=36, step=1, key=f"run_{i}")
    runs_input.append(run)

# Predict button
if st.button("Predict Final Score"):
    if len(runs_input) == current_over:
        try:
            # Convert to cumulative for model
            cumulative_runs = np.cumsum(runs_input).tolist()
            padded_runs = cumulative_runs + [0]*(20 - len(cumulative_runs))
            padded_array = np.array(padded_runs).reshape(-1, 1)

            # Scale & reshape
            scaled_input = scaler_input.transform(padded_array)
            model_input = scaled_input.reshape(1, 20, 1)

            # Predict
            pred_scaled = gru_model.predict(model_input, verbose=0)
            predicted_score = scaler_output.inverse_transform(pred_scaled)[0][0]

            st.success(f"🎯 Predicted Final Score: {predicted_score:.2f} runs")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.warning("⚠️ Please enter runs for all overs.")
