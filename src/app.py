from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------
# Import production pipeline
# ---------------------------------------------------------

from ev_pipeline import EVRangePipeline


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

MODEL_PATH = ROOT / "models" / "ev_range_pipeline.joblib"

if not MODEL_PATH.exists():
    st.error(
        "Production model not found.\n\n"
        "Run src/train.py first to create:\n"
        "models/ev_range_pipeline.joblib"
    )
    st.stop()


predictor = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="⚡",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("⚡ EV Range Predictor")

st.caption(
    "Specification-based prediction of official EV range (km)."
)


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------

with st.form("ev_form"):

    a, b, c = st.columns(3)

    # -----------------------------------------------------
    # Column A
    # -----------------------------------------------------

    with a:

        brand = st.text_input(
            "Brand",
            "Tesla",
        )

        battery = st.number_input(
            "Battery capacity (kWh)",
            min_value=20.0,
            max_value=200.0,
            value=75.0,
        )

        cells = st.number_input(
            "Number of cells",
            min_value=0.0,
            max_value=5000.0,
            value=400.0,
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=50.0,
            max_value=1500.0,
            value=400.0,
        )

        top = st.number_input(
            "Top speed (km/h)",
            min_value=80.0,
            max_value=350.0,
            value=200.0,
        )

        acc = st.number_input(
            "0–100 km/h (s)",
            min_value=2.0,
            max_value=20.0,
            value=5.5,
        )

    # -----------------------------------------------------
    # Column B
    # -----------------------------------------------------

    with b:

        fast = st.number_input(
            "DC fast-charging power (kW)",
            min_value=20.0,
            max_value=500.0,
            value=150.0,
        )

        towing = st.number_input(
            "Towing capacity (kg)",
            min_value=0.0,
            max_value=5000.0,
            value=1000.0,
        )

        cargo = st.number_input(
            "Cargo volume (L)",
            min_value=50.0,
            max_value=3000.0,
            value=450.0,
        )

        seats = st.number_input(
            "Seats",
            min_value=1,
            max_value=9,
            value=5,
        )

        length = st.number_input(
            "Length (mm)",
            min_value=2500.0,
            max_value=6000.0,
            value=4500.0,
        )

        width = st.number_input(
            "Width (mm)",
            min_value=1400.0,
            max_value=2500.0,
            value=1850.0,
        )

    # -----------------------------------------------------
    # Column C
    # -----------------------------------------------------

    with c:

        height = st.number_input(
            "Height (mm)",
            min_value=1200.0,
            max_value=2500.0,
            value=1600.0,
        )

        drive = st.selectbox(
            "Drivetrain",
            ["AWD", "FWD", "RWD"],
        )

        segment = st.selectbox(
            "Segment",
            [
                "B - Compact",
                "C - Medium",
                "D - Large",
                "E - Executive",
                "F - Luxury",
                "JB - Compact",
                "JC - Medium",
                "JD - Large",
                "JF - Luxury",
                "N - Passenger Van",
                "Other",
            ],
        )

        body = st.selectbox(
            "Body type",
            [
                "SUV",
                "Sedan",
                "Hatchback",
                "Small Passenger Van",
                "Liftback Sedan",
                "Station/Estate",
                "Cabriolet",
                "Coupe",
            ],
        )

        port = st.selectbox(
            "Fast-charge port",
            [
                "CCS",
                "CHAdeMO",
                "Missing",
            ],
        )

    # -----------------------------------------------------
    # Submit
    # -----------------------------------------------------

    submitted = st.form_submit_button(
        "🔮 Predict EV Range",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    row = pd.DataFrame(
        [
            {
                "brand": brand,
                "battery_capacity_kWh": battery,
                "number_of_cells": cells,
                "torque_nm": torque,
                "top_speed_kmh": top,
                "acceleration_0_100_s": acc,
                "fast_charging_power_kw_dc": fast,
                "fast_charge_port": port,
                "towing_capacity_kg": towing,
                "cargo_volume_l": cargo,
                "seats": seats,
                "drivetrain": drive,
                "segment": segment,
                "length_mm": length,
                "width_mm": width,
                "height_mm": height,
                "car_body_type": body,
            }
        ]
    )

    try:

        prediction = float(
            predictor.predict(row)[0]
        )

        st.success(
            f"Predicted official range: **{prediction:.0f} km**"
        )

        st.info(
            "efficiency_wh_per_km is intentionally excluded "
            "from the final predictor to avoid target leakage."
        )

    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.exception(error)
