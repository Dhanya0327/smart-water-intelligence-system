import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("data/hostel_water_data.csv")

# ---------------------------------------------------
# SELECT FEATURE
# ---------------------------------------------------

X = df[["daily_factor_Nd"]]

# ---------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------

model = IsolationForest(
    contamination=0.1,
    random_state=42
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model.fit(X)

# ---------------------------------------------------
# PREDICT ANOMALIES
# ---------------------------------------------------

df["Anomaly"] = model.predict(X)

# ---------------------------------------------------
# ANOMALY SCORE
# ---------------------------------------------------

df["Anomaly_Score"] = model.decision_function(X)

# ---------------------------------------------------
# SHOW RESULTS
# ---------------------------------------------------

print("\nANOMALY DETECTION RESULTS:")

print(
    df[
        [
            "academic_year",
            "daily_factor_Nd",
            "Anomaly",
            "Anomaly_Score"
        ]
    ]
)

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
        return "Unusually low consumption"

    else:
        return "Normal behavior pattern"

df["AI_Reason"] = df["daily_factor_Nd"].apply(
    ai_reason
)

# ---------------------------------------------------
# SHOW FINAL OUTPUT
# ---------------------------------------------------

print(
    df[
        [
            "daily_factor_Nd",
            "Anomaly",
            "Risk_Level",
            "AI_Reason"
        ]
    ]
)

# ---------------------------------------------------
# COLORS
# ---------------------------------------------------

colors = [
    'red' if x == -1 else 'blue'
    for x in df["Anomaly"]
]

# ---------------------------------------------------
# GRAPH
# ---------------------------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(
    df.index,
    df["daily_factor_Nd"],
    c=colors,
    s=100
)

plt.title("Isolation Forest AI Detection")

plt.xlabel("Records")

plt.ylabel("Daily Factor")

plt.grid(True)

plt.show()