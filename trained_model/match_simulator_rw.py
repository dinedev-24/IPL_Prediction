import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import random

# Seed for reproducibility
np.random.seed(42)

st.set_page_config(page_title="🏏 IPL Match Simulator", layout="wide")
st.title("🏏 IPL Match Simulator with Team & Points Table Logic")

# -----------------------------
# Match Metadata Section
# -----------------------------
st.sidebar.header("📋 Match Configuration")
teams = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
    "Delhi Capitals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
]
team1 = st.sidebar.selectbox("Select Team 1", teams)
team2 = st.sidebar.selectbox("Select Team 2", [t for t in teams if t != team1])
venue_options = [
    "Wankhede Stadium", "M. A. Chidambaram Stadium", "Eden Gardens",
    "Narendra Modi Stadium", "M. Chinnaswamy Stadium", "Arun Jaitley Stadium",
    "Rajiv Gandhi Intl. Stadium", "Sawai Mansingh Stadium",
    "Ekana Cricket Stadium", "HPCA Stadium"
]
venue = st.sidebar.selectbox("Select Venue", venue_options)
match_date = st.sidebar.date_input("Match Date", datetime.today())
match_status = st.sidebar.selectbox("Match Status", ["Completed", "Live", "Upcoming"])

# -----------------------------
# Match Simulation
# -----------------------------
st.subheader("🎲 Match Simulation (20 Overs)")
overs = list(range(1, 21))
runs_this_over = np.random.randint(0, 21, size=20)
wickets_this_over = np.random.binomial(1, 0.3, size=20)

cumulative_runs = np.cumsum(runs_this_over)
cumulative_wickets = np.cumsum(wickets_this_over)

match_df = pd.DataFrame({
    "Over": overs,
    "Runs_This_Over": runs_this_over,
    "Cumulative_Runs": cumulative_runs,
    "Wickets_This_Over": wickets_this_over,
    "Cumulative_Wickets": cumulative_wickets
})

st.dataframe(match_df, use_container_width=True)

# -----------------------------
# Simulated Match Outcome
# -----------------------------
team1_score = cumulative_runs[-1]
team1_wickets = cumulative_wickets[-1]
team2_score = random.randint(team1_score - 30, team1_score + 30)

if match_status == "Completed":
    winner = team1 if team1_score > team2_score else team2
    st.success(f"🏆 Match Result: **{winner} won!** ({team1}: {team1_score}/{team1_wickets}, {team2}: {team2_score}/?)")
else:
    winner = None
    st.info("Match is still ongoing or upcoming...")

# -----------------------------
# Points Table Simulation
# -----------------------------
st.subheader("📊 Updated Points Table")
team_points = {team: random.randint(4, 16) for team in teams}
if winner:
    team_points[winner] += 2

points_df = pd.DataFrame({
    "Team": list(team_points.keys()),
    "Points": list(team_points.values())
}).sort_values(by="Points", ascending=False).reset_index(drop=True)

st.dataframe(points_df, use_container_width=True)

# -----------------------------
# Predicted Top 4
# -----------------------------
top4 = points_df.head(4)["Team"].tolist()
st.subheader("🔮 Predicted Top 4 Teams")
for idx, t in enumerate(top4, 1):
    st.markdown(f"**{idx}. {t}**")

# -----------------------------
# Match Metadata Display
# -----------------------------
st.markdown("---")
st.markdown(f"""
📍 **Match Info**  
- **Team 1:** {team1}  
- **Team 2:** {team2}  
- **Venue:** {venue}  
- **Date:** {match_date.strftime('%Y-%m-%d')}  
- **Status:** {match_status}
""")