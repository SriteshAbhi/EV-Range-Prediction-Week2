# step2_clean_visualize.py
# -----------------------------------------
# Step 2: Clean the dataset and visualize basic statistics
# -----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("electric_vehicle_data.csv")

# Display missing values before cleaning
print("Missing values before cleaning:\n")
print(df.isnull().sum())

# Fill missing values for each column
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])   # Fill categorical columns with mode
    else:
        df[col] = df[col].fillna(df[col].median())    # Fill numerical columns with median

# Display missing values after cleaning
print("\nMissing values after cleaning:\n")
print(df.isnull().sum())

# -------------------------------
# Basic Data Visualization
# -------------------------------

# 1️⃣ Distribution of Electric Range (now using correct column name: range_km)
plt.figure(figsize=(8, 5))
sns.histplot(df['range_km'], bins=30, kde=True, color='blue')
plt.title("Distribution of Electric Vehicle Range (km)")
plt.xlabel("Range (km)")
plt.ylabel("Frequency")
plt.savefig("EDA_visualizations.png")
plt.show()

# 2️⃣ Top Speed vs Range
plt.figure(figsize=(8, 5))
sns.scatterplot(x='top_speed_kmh', y='range_km', data=df, hue='car_body_type')
plt.title("Top Speed vs Range (by Car Body Type)")
plt.xlabel("Top Speed (km/h)")
plt.ylabel("Range (km)")
plt.legend(title='Car Body Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 3️⃣ Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of EV Numerical Features")
plt.show()
