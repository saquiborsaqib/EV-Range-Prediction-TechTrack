# EV Range Prediction — TechTrack 3.0

## Objective
Predict `range_km` (official driving range in kilometres) from EV specifications.

The final predictor **does not use `efficiency_wh_per_km`**, following the challenge restriction.

## Dataset
- 478 EV models
- 59 brands
- 22 original columns
- Target: `range_km`

## Data cleaning
- Parsed numeric values from `cargo_volume_l`, including the three "Banana Boxes" entries.
- Median/missing-value handling is delegated to CatBoost for numerical fields.
- Missing categorical values are represented as `Missing`.
- No duplicate rows were found.
- Removed constant `battery_type`.
- Removed `source_url` as metadata/identifier.
- Removed near-unique `model` to reduce memorization risk.
- Excluded `efficiency_wh_per_km` from the final model to prevent target leakage.

## Feature engineering
1. `footprint_m2 = length × width`
2. `volume_proxy_m3 = length × width × height`
3. `battery_per_torque`
4. `battery_per_footprint`
5. `performance_index = top_speed / (acceleration + 0.1)`

## Model
Final model: **CatBoostRegressor**

Parameters:
- iterations: 600
- depth: 6
- learning rate: 0.04
- L2 regularization: 5
- random seed: 42

CatBoost was selected because the dataset is small and contains mixed numerical and categorical specifications.

## Evaluation
Fixed holdout: 80/20, `random_state=42`

| Metric | Holdout |
|---|---:|
| MAE | 11.775 km |
| RMSE | 15.104 km |
| R² | 0.9784 |

5-fold shuffled CV mean:

| Metric | CV mean |
|---|---:|
| MAE | 12.995 km |
| RMSE | 18.244 km |
| R² | 0.9679 |

## Run locally

```bash
pip install -r requirements.txt
python src/train.py
streamlit run src/app.py
```

Open the Streamlit URL and enter EV specifications to receive a predicted range.

## Submission contents
- `notebooks/EV_Range_Prediction.ipynb` — end-to-end notebook
- `models/ev_range_catboost.joblib` — trained model
- `models/model_metadata.json` — reproducibility metadata
- `src/train.py` — training script
- `src/app.py` — interactive demo
- `src/pipeline.py` — shared feature engineering
- `submission_report.pdf` — concise technical report
- `data/EvRangePredictionDataset.xlsx` — supplied dataset
