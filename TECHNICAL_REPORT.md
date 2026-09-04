# EV Range Prediction — Competition Technical Report

## 1. Executive Summary
A reproducible regression system was developed to predict official EV range (`range_km`) from static vehicle specifications.

## 2. Dataset
The supplied dataset contains **478 EV models** and **22 original fields**.

## 3. Data Cleaning
- Parsed the three text-form cargo-volume entries into numeric values.
- Retained numerical missing values for CatBoost's native handling.
- Represented missing categorical values as `Missing`.
- Removed constant `battery_type`.
- Removed `source_url` as metadata/identifier.
- Removed near-unique `model` to reduce memorization risk.
- **Excluded `efficiency_wh_per_km` from the final predictor**, as required by the challenge.

## 4. EDA
EDA covered target distribution, numerical relationships, battery-capacity/range relationship and drivetrain-level range summaries. Battery capacity is a strong domain-relevant driver, while other vehicle specifications provide additional information.

## 5. Feature Engineering
Five domain-inspired features were created:
- `footprint_m2`
- `volume_proxy_m3`
- `battery_per_torque`
- `battery_per_footprint`
- `performance_index`

No engineered feature uses the target or `efficiency_wh_per_km`.

## 6. Model Comparison

| Model | MAE (km) | RMSE (km) | R² |
|---|---:|---:|---:|
| CatBoost (Final) | 11.775 | 15.104 | 0.9784 |\n| Ridge Regression | 15.218 | 18.526 | 0.9676 |\n| Random Forest | 14.141 | 19.105 | 0.9655 |\n| Gradient Boosting | 16.144 | 20.688 | 0.9596 |\n
**Selection:** CatBoost (Final) achieved the lowest holdout RMSE and strongest overall holdout performance. CatBoost was selected because it is technically well suited to small mixed-type tabular data.

## 7. Cross-Validation
Five-fold shuffled CV for the final model:
- MAE: **12.995 km**
- RMSE: **18.244 km**
- R²: **0.9679**

## 8. Explainability
Feature importance is used to inspect model behaviour. It is not causal inference.

## 9. Reproducibility
The submission contains a competition-ready notebook, saved model, reusable pipeline, metadata, training code, requirements and Streamlit app. Random seed is fixed at 42.

## 10. Limitations
This is a specification-based predictor, not a real-time estimator. The dataset lacks weather, traffic, HVAC, State of Charge, State of Health and trip telemetry.

## 11. Live Demo
Launch Streamlit, enter EV specifications, click **Predict range**, then vary inputs to demonstrate immediate response through the saved pipeline.
