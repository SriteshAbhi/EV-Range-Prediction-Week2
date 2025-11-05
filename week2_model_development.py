# week2_model_development.py
# Electric Vehicle Range Prediction - Model Development (Week 2)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Step 1: Load Dataset
try:
    data = pd.read_csv("electric_vehicle_data.csv")
    print("✅ Data loaded successfully\n")
except Exception as e:
    print("Error loading data:", e)

# Display first few rows
print("Sample Data:\n", data.head(), "\n")
print("Columns in dataset:\n", data.columns.tolist(), "\n")

# Step 2: Selecting features and target
X = data[['battery_capacity_kWh', 'top_speed_kmh', 'efficiency_wh_per_km']]
y = data['range_km']

# Step 3: Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

print("🔍 Model Evaluation Results:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = [mse, r2]

    print(f"Model: {name}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R² Score: {r2:.3f}\n")

# Step 5: Compare models
results_df = pd.DataFrame(results, index=['MSE', 'R2']).T
print("\n📊 Final Model Comparison:\n")
print(results_df)

# Step 6: Plot model performance
plt.bar(results_df.index, results_df['R2'], color=['skyblue', 'lightgreen', 'orange'])
plt.title("Model Performance Comparison (R2 Score)")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig("model_performance.png")
plt.show()

# Step 7: Save the best model
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]
joblib.dump(best_model, "best_ev_model.pkl")

print(f"\n💾 Best model '{best_model_name}' saved successfully as 'best_ev_model.pkl'")

print("\n✅ Week 2 Model Development Completed Successfully!")
