from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from ev_pipeline import EVRangePipeline


DATA = ROOT / "data" / "EvRangePredictionDataset.xlsx"

MODEL_PATH = (
    ROOT
    / "models"
    / "ev_range_pipeline.joblib"
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("=" * 60)
print("EV RANGE PREDICTION — TRAINING")
print("=" * 60)

df = pd.read_excel(DATA)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

target = df["range_km"]


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
)

y_train = train_df["range_km"]
y_test = test_df["range_km"]


# ---------------------------------------------------------
# Train production pipeline
# ---------------------------------------------------------

pipeline = EVRangePipeline()

pipeline.fit(
    train_df,
    y_train,
)


# ---------------------------------------------------------
# Test prediction
# ---------------------------------------------------------

test_input = test_df.drop(
    columns=["range_km"],
    errors="ignore",
)

pred = pipeline.predict(
    test_input
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    pred,
)

rmse = mean_squared_error(
    y_test,
    pred,
) ** 0.5

r2 = r2_score(
    y_test,
    pred,
)


print()
print("HOLDOUT RESULTS")
print("-" * 40)

print(f"MAE : {mae:.3f} km")
print(f"RMSE: {rmse:.3f} km")
print(f"R²  : {r2:.4f}")


# ---------------------------------------------------------
# Refit on ALL data for deployment
# ---------------------------------------------------------

print()
print("Training final production model on all data...")

final_pipeline = EVRangePipeline()

final_pipeline.fit(
    df,
    target,
)


# ---------------------------------------------------------
# Save production artifact
# ---------------------------------------------------------

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    final_pipeline,
    MODEL_PATH,
)


print()
print("✅ Production model saved:")
print(MODEL_PATH)


# ---------------------------------------------------------
# Serialization verification
# ---------------------------------------------------------

del final_pipeline

loaded_pipeline = joblib.load(
    MODEL_PATH
)

print()
print("SERIALIZATION CHECK")
print("-" * 40)

print(
    "Loaded type:",
    type(loaded_pipeline),
)

print(
    "Loaded module:",
    type(loaded_pipeline).__module__,
)

assert isinstance(
    loaded_pipeline,
    EVRangePipeline,
)

assert (
    type(loaded_pipeline).__module__
    == "ev_pipeline"
)

print()
print("🎉 JOBLIB SERIALIZATION CHECK PASSED")
print("🎉 TRAINING COMPLETE")
