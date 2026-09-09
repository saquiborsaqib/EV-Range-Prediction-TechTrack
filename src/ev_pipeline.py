import re

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


class EVRangePipeline:
    """
    Production preprocessing + feature engineering + CatBoost model
    for EV driving-range prediction.
    """

    def __init__(self, params=None):
        self.params = params or {
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.04,
            "loss_function": "RMSE",
            "l2_leaf_reg": 5,
            "random_seed": 42,
            "verbose": False,
        }

        self.model = CatBoostRegressor(**self.params)
        self.columns_ = None
        self.cats_ = None

    @staticmethod
    def _cargo_to_numeric(value):
        if pd.isna(value):
            return np.nan

        match = re.search(r"\d+(?:\.\d+)?", str(value))

        if match:
            return float(match.group())

        return np.nan

    def transform(self, X):
        data = X.copy()

        # Convert cargo volume to numeric.
        if "cargo_volume_l" in data.columns:
            data["cargo_volume_l"] = data["cargo_volume_l"].apply(
                self._cargo_to_numeric
            )

        # Remove target/leakage/non-predictive columns.
        data = data.drop(
            columns=[
                "range_km",
                "efficiency_wh_per_km",
                "source_url",
                "model",
                "battery_type",
            ],
            errors="ignore",
        )

        # Domain feature engineering.
        data["footprint_m2"] = (
            data["length_mm"] * data["width_mm"] / 1_000_000
        )

        data["volume_proxy_m3"] = (
            data["length_mm"]
            * data["width_mm"]
            * data["height_mm"]
            / 1_000_000_000
        )

        data["battery_per_torque"] = (
            data["battery_capacity_kWh"]
            / (data["torque_nm"].abs() + 1)
        )

        data["battery_per_footprint"] = (
            data["battery_capacity_kWh"]
            / (data["footprint_m2"] + 1e-6)
        )

        data["performance_index"] = (
            data["top_speed_kmh"]
            / (data["acceleration_0_100_s"] + 0.1)
        )

        # If model schema is already known, enforce it.
        if self.columns_ is not None:

            for column in self.columns_:
                if column not in data.columns:
                    data[column] = np.nan

            data = data[self.columns_]

        # CatBoost categorical columns cannot contain NaN.
        categorical = data.select_dtypes(
            include=["object"]
        ).columns.tolist()

        for column in categorical:
            data[column] = (
                data[column]
                .fillna("Missing")
                .astype(str)
            )

        return data

    def fit(self, X, y):

        transformed = self.transform(X)

        # Save exact training feature schema.
        self.columns_ = transformed.columns.tolist()

        # Transform again so the saved schema is enforced.
        transformed = self.transform(X)

        self.cats_ = transformed.select_dtypes(
            include=["object"]
        ).columns.tolist()

        cat_indices = [
            transformed.columns.get_loc(column)
            for column in self.cats_
        ]

        self.model.fit(
            transformed,
            y,
            cat_features=cat_indices,
        )

        return self

    def predict(self, X):
        transformed = self.transform(X)

        return self.model.predict(transformed)
