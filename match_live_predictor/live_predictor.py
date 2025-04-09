# live_predictor.py

import streamlit as st
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# App config
st.set_page_config(page_title="🏏 IPL Live Match Predictor", layout="wide")
st.title("🏏 IPL Live Match Predictor")

# Match Setup
st.header("📋 Match Setup")
col1, col2, col3 = st.columns(3)

with col1:
    team_a = st.selectbox("🏏 Team A", ["CSK", "MI", "RCB", "GT", "RR", "LSG", "KKR", "SRH"])
with col2:
    team_b = st.selectbox("🏏 Team B", ["CSK", "MI", "RCB", "GT", "RR", "LSG", "KKR", "SRH"])
with col3:
    venue = st.selectbox("📍 Venue", ["Wankhede Stadium", "Chepauk", "Eden Gardens", "Narendra Modi Stadium", "Chinnaswamy Stadium"])

match_date = st.date_input("📅 Match Date", datetime.date.today())
match_status = st.radio("📶 Match Status", ["Scheduled", "In Progress", "Completed"])

# Section: Simulate Match Data
st.header("🎲 Simulate Over-wise Runs & Wickets")

if st.button("🔁 Generate Match"):
    overs = list(range(1, 21))
    runs_this_over = np.random.randint(0, 21, size=20)  # realistic run range
    wickets_this_over = np.random.binomial(1, 0.3, size=20)

    cumulative_runs = np.cumsum(runs_this_over)
    cumulative_wickets = np.cumsum(wickets_this_over)

    df = pd.DataFrame({
        "Over": overs,
        "Runs_This_Over": runs_this_over,
        "Cumulative_Runs": cumulative_runs,
        "Wickets_This_Over": wickets_this_over,
        "Cumulative_Wickets": cumulative_wickets
    })

    st.session_state["live_df"] = df
    st.success(f"📝 {team_a} vs {team_b} at {venue} on {match_date}")
    st.dataframe(df, use_container_width=True)

# GRU Prediction
if "live_df" in st.session_state:
    df = st.session_state["live_df"]
    st.subheader("📈 Predicted Final Score (GRU Model)")

    try:
        # Load model and scalers
        model_path = os.path.join("match_live_predictor", "gru_score_predictor_rw.keras")
        input_scaler_path = os.path.join("match_live_predictor", "scaler_input_rw.save")
        output_scaler_path = os.path.join("match_live_predictor", "scaler_output_rw.save")

        model = load_model(model_path, compile=False)
        scaler_input = joblib.load(input_scaler_path)
        scaler_output = joblib.load(output_scaler_path)

        # Prepare input
        match_input = df[["Runs_This_Over", "Wickets_This_Over"]].values
        padded_input = np.pad(match_input, ((0, 20 - len(match_input)), (0, 0)), mode="constant")

        scaled_input = scaler_input.transform(padded_input)
        reshaped_input = scaled_input.reshape(1, 20, 2)

        # Predict final score
        pred_scaled = model.predict(reshaped_input, verbose=0)
        predicted_final = scaler_output.inverse_transform(pred_scaled)[0][0]

        # Display predicted score
        st.success(f"🎯 Predicted Final Score: {predicted_final:.2f} runs")

        # Visualization
        fig, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(df["Over"], df["Cumulative_Runs"], marker='o', label="Cumulative Runs", color="green")
        ax1.set_xlabel("Overs")
        ax1.set_ylabel("Runs", color="green")
        ax1.tick_params(axis='y', labelcolor="green")
        ax1.set_title("📊 Match Progression & GRU Prediction")

        ax2 = ax1.twinx()
        ax2.bar(df["Over"], df["Wickets_This_Over"], alpha=0.3, color="red", label="Wickets")
        ax2.set_ylabel("Wickets", color="red")
        ax2.tick_params(axis='y', labelcolor="red")

        plt.axhline(y=predicted_final, color="blue", linestyle="--", label="Predicted Final")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.85))
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Prediction module error: {e}")

# Footer
st.markdown("""
---
👨‍💻 Built by Dinesh Kumar | Powered by Streamlit, GRU, NumPy
""")
