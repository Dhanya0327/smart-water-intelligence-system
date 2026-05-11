import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import numpy as np

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Water Intelligence System",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("AI-Powered Smart Hostel Water Intelligence System")

st.markdown(
    """
This platform uses Artificial Intelligence and
Isolation Forest Machine Learning to analyze
hostel water usage patterns, detect anomalies,
predict suspicious behavior, and generate
sustainability insights.
"""
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("data/hostel_water_data.csv")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Project Information")

st.sidebar.info(
    """
Features Included:

- Isolation Forest AI
- Anomaly Detection
- Risk Classification
- Explainable AI
- Sustainability Analytics
- Interactive Visualization
- Smart AI Insights
"""
)

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df)

# ---------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------

X = df[["daily_factor_Nd"]]

# ---------------------------------------------------
# CREATE AI MODEL
# ---------------------------------------------------

model = IsolationForest(
    contamination=0.1,
    random_state=42
)

# ---------------------------------------------------
# TRAIN & PREDICT
# ---------------------------------------------------

df["Anomaly"] = model.fit_predict(X)

df["Anomaly_Score"] = model.decision_function(X)

# ---------------------------------------------------
# RISK LEVEL
# ---------------------------------------------------

def classify_risk(score):

    if score < -0.15:
        return "Critical"

    elif score < -0.08:
        return "High"

    elif score < -0.03:
        return "Medium"

    else:
        return "Low"

df["Risk_Level"] = df["Anomaly_Score"].apply(
    classify_risk
)

# ---------------------------------------------------
# AI REASON
# ---------------------------------------------------

mean_value = df["daily_factor_Nd"].mean()

def ai_reason(value):

    if value > mean_value * 1.5:
        return "Possible abnormal high usage"

    elif value < mean_value * 0.5:
        return "Unusually low usage"

    else:
        return "Normal behavior pattern"

df["AI_Reason"] = df["daily_factor_Nd"].apply(
    ai_reason
)

# ---------------------------------------------------
# LEAKAGE PREDICTION
# ---------------------------------------------------

def leakage_prediction(value):

    if value > mean_value * 1.8:
        return "Possible Major Leakage"

    elif value > mean_value * 1.3:
        return "Moderate Leakage Risk"

    else:
        return "Normal"

df["Leakage_Status"] = df["daily_factor_Nd"].apply(
    leakage_prediction
)

# ---------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------

st.subheader("AI Detection Results")

st.dataframe(
    df[
        [
            "academic_year",
            "daily_factor_Nd",
            "Anomaly",
            "Anomaly_Score",
            "Risk_Level",
            "AI_Reason",
            "Leakage_Status"
        ]
    ]
)

# ---------------------------------------------------
# FILTER ANOMALIES
# ---------------------------------------------------

anomalies = df[df["Anomaly"] == -1]

# ---------------------------------------------------
# DETECTED ANOMALIES
# ---------------------------------------------------

st.subheader("Detected Anomalies")

st.dataframe(anomalies)

# ---------------------------------------------------
# INTERACTIVE AI GRAPH
# ---------------------------------------------------

st.subheader("Interactive AI Visualization")

fig = px.scatter(
    df,
    x=df.index,
    y="daily_factor_Nd",
    color="Risk_Level",
    size=abs(df["Anomaly_Score"]),
    hover_data=[
        "academic_year",
        "AI_Reason",
        "Leakage_Status"
    ],
    title="Isolation Forest AI Detection"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# LINE GRAPH
# ---------------------------------------------------

st.subheader("Water Consumption Trend")

line_fig = px.line(
    df,
    y="daily_factor_Nd",
    title="Daily Water Consumption Trend"
)

st.plotly_chart(
    line_fig,
    use_container_width=True
)

# ---------------------------------------------------
# SUSTAINABILITY SCORE
# ---------------------------------------------------

st.subheader("Hostel Sustainability Score")

average_usage = df["daily_factor_Nd"].mean()

eco_score = max(
    0,
    100 - average_usage * 10
)

st.metric(
    "Eco Score",
    f"{eco_score:.2f}%"
)

# ---------------------------------------------------
# LIVE AI MONITORING
# ---------------------------------------------------

st.subheader("Live Water Monitoring")

live_value = np.random.normal(
    mean_value,
    0.2
)

st.metric(
    "Simulated Live Water Usage",
    f"{live_value:.2f}"
)

# ---------------------------------------------------
# AI ALERTS
# ---------------------------------------------------

st.subheader("AI Alerts")

if len(anomalies) > 0:

    for index, row in anomalies.iterrows():

        st.error(
            f"""
Possible abnormal water usage detected.

Academic Year:
{row['academic_year']}

Risk Level:
{row['Risk_Level']}

Leakage Status:
{row['Leakage_Status']}

AI Reason:
{row['AI_Reason']}
"""
        )

else:

    st.success(
        "No major anomalies detected."
    )

# ---------------------------------------------------
# SMART AI INSIGHTS
# ---------------------------------------------------

st.subheader("Smart AI Insights")

max_value = df["daily_factor_Nd"].max()

min_value = df["daily_factor_Nd"].min()

st.info(
    f"""
Average Daily Factor:
{mean_value:.2f}

Maximum Daily Factor:
{max_value:.2f}

Minimum Daily Factor:
{min_value:.2f}

The AI system continuously analyzes
hostel water behavior and identifies
unusual patterns that may indicate:

- Leakage
- Abnormal consumption
- Sustainability concerns
- Irregular hostel behavior
"""
)

# ---------------------------------------------------
# PROJECT SUMMARY
# ---------------------------------------------------

st.subheader("Project Summary")

st.write(
    """
This project combines:

- Artificial Intelligence
- Isolation Forest Anomaly Detection
- Explainable AI
- Sustainability Intelligence
- Leakage Prediction
- Statistical Analysis
- Interactive Visualization

The system helps hostel administrators:

- detect suspicious water usage
- identify possible leakage
- monitor sustainability
- understand water behavior patterns

This creates an advanced AI-powered
smart water intelligence platform.
"""
)