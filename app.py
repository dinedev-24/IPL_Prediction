import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# ===============================
# Sidebar Navigation
# ===============================
st.set_page_config(page_title="🏏 IPL Unified Predictor App", layout="wide")
selected_tab = st.sidebar.selectbox("🏆 Select Module", [
    "Dashboard",
    "Commentary",
    "Scenario Simulator",
    "Match Simulator",
    "Live Match Predictor"
])

# ===============================
# 1. Dashboard Intro
# ===============================
if selected_tab == "Dashboard":
    st.title("📊 IPL Unified Prediction & Simulation Dashboard")
    st.markdown("""
    Welcome to the complete IPL analytics suite. This unified app supports:
    - 🎯 Score Prediction using GRU
    - 🧠 GPT-3.5 Commentary
    - 🎲 Match Scenario Simulator
    - 📈 Real-time Momentum & Points Table
    - 🧪 Live Match Score Prediction
    
    Use the sidebar to navigate through modules.
    """)
    st.image("match_momentum_dashboard.gif", use_column_width=True)

# ===============================
# 2. Commentary Dashboard
# ===============================
elif selected_tab == "Commentary":
    st.title("🧠 Score Prediction & GPT Commentary")
    try:
        df = pd.read_csv("gru_match_simulation_commentary.csv")
        st.dataframe(df)
        st.markdown("""
        **Commentary Summary:**
        > The team started steadily, losing early wickets in the powerplay. However, they built partnerships during middle overs and accelerated towards the end with a flurry of boundaries.
        """)
        st.image("gru_match_simulation_plot.gif", caption="Score + Commentary Summary")
    except:
        st.warning("Upload commentary file or check dataset path.")

# ===============================
# 3. Scenario Simulator
# ===============================
elif selected_tab == "Scenario Simulator":
    st.title("🎯 Match Scenario Score Predictor")
    st.markdown("Enter cumulative runs over 20 overs to predict the final score.")

    current_over = st.slider("Select Overs Completed", 1, 20, 10)
    runs_input = []

    for i in range(current_over):
        run = st.number_input(f"Runs after Over {i+1}", min_value=0, step=1, key=f"r_{i}")
        runs_input.append(run)

    if st.button("Predict Score"):
        if len(runs_input) == current_over:
            padded = runs_input + [0]*(20 - len(runs_input))
            arr = np.array(padded).reshape(-1, 1)
            model = load_model("gru_score_predictor.keras", compile=False)
            scaler_input = joblib.load("scaler_input.save")
            scaler_output = joblib.load("scaler_output.save")

            scaled = scaler_input.transform(arr).reshape(1, 20, 1)
            pred = model.predict(scaled)
            predicted_score = scaler_output.inverse_transform(pred)[0][0]

            st.success(f"🏏 Predicted Final Score: {predicted_score:.2f} runs")

# ===============================
# 4. Match Simulator + Points Table
# ===============================
elif selected_tab == "Match Simulator":
    st.title("🏟️ IPL Match Simulator + Points Table")
    teams = ["CSK", "MI", "RCB", "GT", "RR", "LSG", "KKR", "SRH"]
    match_results = []
    points = {team: 0 for team in teams}

    for i in range(10):
        t1, t2 = np.random.choice(teams, 2, replace=False)
        winner = np.random.choice([t1, t2])
        points[winner] += 2
        match_results.append((t1, t2, winner))

    df_matches = pd.DataFrame(match_results, columns=["Team A", "Team B", "Winner"])
    df_points = pd.DataFrame(sorted(points.items(), key=lambda x: x[1], reverse=True), columns=["Team", "Points"])

    st.subheader("📝 Simulated Match Results")
    st.dataframe(df_matches)

    st.subheader("📊 Points Table")
    st.dataframe(df_points)

# ===============================
# 5. Live Match Predictor (Runs + Wickets)
# ===============================
elif selected_tab == "Live Match Predictor":
    st.title("📡 IPL Live Match Predictor")

    col1, col2, col3 = st.columns(3)
    with col1:
        team_a = st.selectbox("Team A", ["MI", "RCB", "CSK", "GT"])
    with col2:
        team_b = st.selectbox("Team B", ["MI", "RCB", "CSK", "GT"])
    with col3:
        venue = st.selectbox("Venue", ["Wankhede", "Chinnaswamy", "Chepauk"])

    date = st.date_input("Match Date", datetime.date.today())
    status = st.radio("Match Status", ["In Progress", "Completed", "Scheduled"])

    if st.button("Generate & Predict"):
        runs = np.random.randint(0, 21, 20)
        wickets = np.random.binomial(1, 0.3, 20)

        df = pd.DataFrame({
            "Over": list(range(1, 21)),
            "Runs_This_Over": runs,
            "Cumulative_Runs": np.cumsum(runs),
            "Wickets_This_Over": wickets,
            "Cumulative_Wickets": np.cumsum(wickets)
        })

        st.success(f"Match: {team_a} vs {team_b} at {venue} on {date}")
        st.dataframe(df)

        # Prediction logic
        model = load_model("gru_score_predictor_rw.keras", compile=False)
        scaler_input = joblib.load("scaler_rw_input.save")
        scaler_output = joblib.load("scaler_rw_output.save")

        padded_input = np.pad(df[["Runs_This_Over", "Wickets_This_Over"]].values,
                              ((0, 20 - df.shape[0]), (0, 0)), mode='constant')
        scaled_input = scaler_input.transform(padded_input).reshape(1, 20, 2)
        pred_scaled = model.predict(scaled_input)
        predicted_score = scaler_output.inverse_transform(pred_scaled)[0][0]

        st.success(f"🎯 Predicted Final Score: {predicted_score:.2f} runs")

        fig, ax1 = plt.subplots()
        ax1.plot(df["Over"], df["Cumulative_Runs"], marker='o', color='green', label='Runs')
        ax1.set_xlabel("Over")
        ax1.set_ylabel("Runs", color='green')
        ax2 = ax1.twinx()
        ax2.bar(df["Over"], df["Wickets_This_Over"], alpha=0.4, color='red', label='Wickets')
        ax2.set_ylabel("Wickets", color='red')
        plt.axhline(y=predicted_score, color='blue', linestyle='--', label='Predicted Score')
        fig.legend(loc='upper left')
        st.pyplot(fig)

# ===============================
# Footer
# ===============================
st.markdown("""
---
👨‍💻 Built by Dinesh Kumar | Powered by Streamlit, NumPy, Matplotlib, TensorFlow
""")
