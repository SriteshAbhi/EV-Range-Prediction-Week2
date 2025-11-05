# EV Range Prediction – Week 2

In this week, I developed machine learning models to predict the range of electric vehicles using real EV specifications data.

## Tasks Done:
- Used data from Week 1 (cleaned and processed dataset)
- Built three ML models:  
  - Linear Regression  
  - Decision Tree  
  - Random Forest
- Compared model performances using MSE and R² Score
- Visualized model comparison in a bar graph

## Results:
- **Linear Regression:** R² = 0.882  
- **Decision Tree:** R² = 0.91  
- **Random Forest:** R² = 0.93 (Best performing model)

## Files Included:
- `week2_model_development.py` – Model building and evaluation
- `model_performance.png` – Performance comparison chart
- `model_results.csv` – MSE and R² results
- `best_ev_model.pkl` – Saved trained model
- `electric_vehicle_data.csv` – Dataset
- `step2_clean_visualize.py` – Data cleaning & visualization script

## Summary:
The Random Forest model performed best, showing high accuracy in predicting EV range.
