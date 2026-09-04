from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.pipeline import EVFeatureEngineer, build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "EvRangePredictionDataset.xlsx"
MODEL_PATH = ROOT / "models" / "ev_range_catboost.joblib"

df = pd.read_excel(DATA)
target = df["range_km"]
X = EVFeatureEngineer().fit_transform(df) if hasattr(EVFeatureEngineer, "fit_transform") else EVFeatureEngineer().fit(df).transform(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.20, random_state=42
)

categorical = X.select_dtypes(include=["object"]).columns.tolist()
model = build_model()
model.fit(
    X_train,
    y_train,
    cat_features=[X_train.columns.get_loc(c) for c in categorical],
)

pred = model.predict(X_test)
print(f"MAE : {mean_absolute_error(y_test, pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.3f}")
print(f"R2  : {r2_score(y_test, pred):.4f}")

# Refit on all data for deployment.
final_model = build_model()
final_model.fit(
    X, target,
    cat_features=[X.columns.get_loc(c) for c in categorical],
)
joblib.dump(final_model, MODEL_PATH)
print(f"Saved: {MODEL_PATH}")
