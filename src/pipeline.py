import re
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

class EVFeatureEngineer:
    """Clean raw EV specifications and create domain-inspired features."""
    def fit(self, X, y=None):
        return self

    @staticmethod
    def _cargo_to_numeric(value):
        if pd.isna(value):
            return np.nan
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(match.group()) if match else np.nan

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        data = X.copy()
        if "cargo_volume_l" in data.columns:
            data["cargo_volume_l"] = data["cargo_volume_l"].apply(
                self._cargo_to_numeric
            )

        drop_cols = [
            "range_km",
            "efficiency_wh_per_km",
            "source_url",
            "model",
            "battery_type",
        ]
        data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")

        data["footprint_m2"] = (
            data["length_mm"] * data["width_mm"] / 1_000_000
        )
        data["volume_proxy_m3"] = (
            data["length_mm"] * data["width_mm"] * data["height_mm"]
            / 1_000_000_000
        )
        data["battery_per_torque"] = (
            data["battery_capacity_kWh"] / (data["torque_nm"].abs() + 1)
        )
        data["battery_per_footprint"] = (
            data["battery_capacity_kWh"] / (data["footprint_m2"] + 1e-6)
        )
        data["performance_index"] = (
            data["top_speed_kmh"] / (data["acceleration_0_100_s"] + 0.1)
        )

        categorical = data.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical:
            data[col] = data[col].fillna("Missing").astype(str)
        return data

def build_model():
    return CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.04,
        loss_function="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
    )
