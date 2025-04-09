\-\-- title: \"🏏 IPL Match Predictor & Simulator\" emoji: \"🏏\"
colorFrom: \"yellow\" colorTo: \"blue\" sdk: \"streamlit\" sdk_version:
\"1.30.0\" app_file: \"app.py\" pinned: false license: \"mit\" tags:  -
cricket  - ipl  - deep-learning  - streamlit  - gru  - prediction  -
sports  - commentary \-\--

\# 🏏 IPL Match Predictor & Live Simulation App

Welcome to the \*\*IPL Match Predictor\*\* -- a powerful and interactive
Streamlit-based application that simulates and predicts IPL match
outcomes using machine learning, deep learning, and Generative AI
commentary.

\-\--

\## 🚀 Key Features

\### 🎯 1. IPL Score Prediction Dashboard (GRU-Based) - Predicts final
match score using a GRU deep learning model. - Inputs: Over-wise runs
and optional wickets (20 overs). - Model trained on synthetic match data
to mimic real T20 dynamics.

\### 🧠 2. GPT Commentary Generator (RAG-based, Optional) - Generate
multi-turn AI commentary based on match progression. - Integrates
GPT-3.5 (OpenAI) with retrieval-based chunked summaries. \*(Optional
deployment upgrade)\*

\### 🧮 3. Match Scenario Simulator - Simulate custom match scenarios
based on user-input cumulative scores. - Get final score predictions
even with mid-match inputs.

\### 📊 4. IPL Match Simulation + Points Table Logic - Team vs Team
match generation with venue and status (Completed/In
Progress/Scheduled). - Dynamic scorecard visualization with runs,
wickets, and predicted outcomes.

\### 📈 5. Live Match Predictor - Simulate full match: generates random
over-wise runs + wickets. - Visualizes match progression and predicts
final score with model confidence.

\-\--

\## 📁 Project Structure

final_app/ │ ├── app.py \# ✅ Unified Streamlit app ├── requirements.txt
\# ✅ Dependencies for Hugging Face ├── trained_model/ \# ✅ GRU-based
models + scalers ├── match_live_predictor/ \# ✅ Live match predictor
files ├── match_simulator.py \# Scenario-based simulator ├── \*.h5 /
.keras \# DL models (GRU, BiLSTM, CNN, LSTM) ├── scaler\_.save \#
Input/output scalers ├── \*.csv \# Match data for commentary, points,
visuals ├── \*.gif \# Dashboard animations └── \*.ipynb \# Phase
notebooks and analysis

\-\--

\## 🔧 Requirements

Install dependencies using:

\`\`\`bash pip install -r requirements.txt

Model Info Model: GRU-based sequence predictor

Input: 20 timesteps (Runs, Wickets)

Output: Scaled prediction of final score

Scaler: MinMaxScaler fitted on input & target

GPT Commentary (Optional) To enable RAG-based GPT-3.5 commentary:

Set your OPENAI_API_KEY in Hugging Face secrets.

Add commentary loader and chunker for match summaries.

🧑‍💻 Built By 👨‍💻 Dinesh Kumar \| Powered by Streamlit, TensorFlow, NumPy,
Hugging Face, and Ope
