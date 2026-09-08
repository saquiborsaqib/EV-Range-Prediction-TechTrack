from pathlib import Path
import sys
import joblib, pandas as pd, streamlit as st

ROOT=Path(__file__).resolve().parents[1]
predictor=joblib.load(ROOT/"models"/"ev_range_pipeline.joblib")
st.set_page_config(page_title="EV Range Predictor",page_icon="⚡",layout="wide")
st.title("⚡ EV Range Predictor")
st.caption("Specification-based prediction of official EV range (km).")

with st.form("ev_form"):
    a,b,c=st.columns(3)
    with a:
        brand=st.text_input("Brand","Tesla")
        battery=st.number_input("Battery capacity (kWh)",20.,200.,75.)
        cells=st.number_input("Number of cells",0.,5000.,400.)
        torque=st.number_input("Torque (Nm)",50.,1500.,400.)
        top=st.number_input("Top speed (km/h)",80.,350.,200.)
        acc=st.number_input("0–100 km/h (s)",2.,20.,5.5)
    with b:
        fast=st.number_input("DC fast-charging power (kW)",20.,500.,150.)
        towing=st.number_input("Towing capacity (kg)",0.,5000.,1000.)
        cargo=st.number_input("Cargo volume (L)",50.,3000.,450.)
        seats=st.number_input("Seats",1,9,5)
        length=st.number_input("Length (mm)",2500.,6000.,4500.)
        width=st.number_input("Width (mm)",1400.,2500.,1850.)
    with c:
        height=st.number_input("Height (mm)",1200.,2500.,1600.)
        drive=st.selectbox("Drivetrain",["AWD","FWD","RWD"])
        segment=st.selectbox("Segment",["B - Compact","C - Medium","D - Large","E - Executive","F - Luxury","JB - Compact","JC - Medium","JD - Large","JF - Luxury","N - Passenger Van","Other"])
        body=st.selectbox("Body type",["SUV","Sedan","Hatchback","Small Passenger Van","Liftback Sedan","Station/Estate","Cabriolet","Coupe"])
        port=st.selectbox("Fast-charge port",["CCS","CHAdeMO","Missing"])
    submitted=st.form_submit_button("Predict range",type="primary")

if submitted:
    row=pd.DataFrame([{"brand":brand,"battery_capacity_kWh":battery,"number_of_cells":cells,"torque_nm":torque,"top_speed_kmh":top,"acceleration_0_100_s":acc,"fast_charging_power_kw_dc":fast,"fast_charge_port":port,"towing_capacity_kg":towing,"cargo_volume_l":cargo,"seats":seats,"drivetrain":drive,"segment":segment,"length_mm":length,"width_mm":width,"height_mm":height,"car_body_type":body}])
    prediction=float(predictor.predict(row)[0])
    st.success(f"Predicted official range: **{prediction:.0f} km**")
    st.info("efficiency_wh_per_km is intentionally not used by the final predictor.")
