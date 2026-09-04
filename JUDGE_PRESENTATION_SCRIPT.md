# 🎤 Judge Presentation Script — 3–5 Minutes

## 1. Problem
“Our goal is to predict official EV range in kilometres from static vehicle specifications.”

## 2. Data
“The supplied dataset has 478 EV models with battery, performance, charging, dimensions, drivetrain, body and other specifications.”

## 3. Cleaning
“We audited missing values, duplicates, types and unusual cargo entries. We parsed the cargo anomaly, handled missing values appropriately, removed constant battery type, removed URL metadata and excluded near-unique model names.”

## 4. Leakage Prevention
“The critical restriction is that efficiency_wh_per_km must not be a final model input because it is directly related to range. We use it only for EDA or sanity checks.”

## 5. Feature Engineering
“We created footprint, volume proxy, battery-per-torque, battery-per-footprint and performance-index features to capture physical and performance relationships.”

## 6. Model Selection
“We compared Ridge, Random Forest, Gradient Boosting and CatBoost on the same holdout split.”
- CatBoost (Final): MAE 11.8 km, RMSE 15.1 km, R² 0.978\n- Ridge Regression: MAE 15.2 km, RMSE 18.5 km, R² 0.968\n- Random Forest: MAE 14.1 km, RMSE 19.1 km, R² 0.966\n- Gradient Boosting: MAE 16.1 km, RMSE 20.7 km, R² 0.960\n
“CatBoost was selected because it gave the strongest result and handles mixed tabular data well.”

## 7. Validation
“Five-fold cross-validation gives a mean MAE of 13.0 km and R² of 0.968, giving us a stability check beyond one split.”

## 8. Live Demo
“I’ll now change the EV specifications and show that the saved pipeline transforms the inputs and returns a prediction immediately.”

## 9. Limitation
“This is a specification-based predictor, not a real-time driving-range estimator. Weather, traffic, HVAC and battery-state variables are not present in the supplied data.”

## Closing
“Our focus is a complete, leakage-safe, reproducible workflow from raw EV specifications to a deployable interactive prediction system.”
