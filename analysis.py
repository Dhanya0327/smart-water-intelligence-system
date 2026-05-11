import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("data/hostel_water_data.csv")

# ---------------------------------------------------
# FIRST 5 ROWS
# ---------------------------------------------------

print("\nFIRST 5 ROWS:")
print(df.head())

# ---------------------------------------------------
# DATASET INFO
# ---------------------------------------------------

print("\nDATASET INFORMATION:")
print(df.info())

# ---------------------------------------------------
# COLUMN NAMES
# ---------------------------------------------------

print("\nCOLUMN NAMES:")
print(df.columns)

# ---------------------------------------------------
# STATISTICAL SUMMARY
# ---------------------------------------------------

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# ---------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ---------------------------------------------------
# MEAN VALUE
# ---------------------------------------------------

mean_value = df["daily_factor_Nd"].mean()

print("\nAVERAGE DAILY FACTOR:")
print(mean_value)

# ---------------------------------------------------
# MAXIMUM VALUE
# ---------------------------------------------------

max_value = df["daily_factor_Nd"].max()

print("\nMAXIMUM DAILY FACTOR:")
print(max_value)

# ---------------------------------------------------
# MINIMUM VALUE
# ---------------------------------------------------

min_value = df["daily_factor_Nd"].min()

print("\nMINIMUM DAILY FACTOR:")
print(min_value)

# ---------------------------------------------------
# GRAPH
# ---------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["daily_factor_Nd"],
    marker='o'
)

plt.title("Daily Water Consumption Trend")

plt.xlabel("Records")

plt.ylabel("Daily Factor")

plt.grid(True)

plt.show()