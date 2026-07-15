"""
Step 9: Deploy - a web app for the House Price Predictor.

The user moves sliders (income, location, rooms...) and the trained
Random Forest instantly predicts the house price.

Run from the project folder with:  py -m streamlit run app.py
Then open http://localhost:8501 in your browser.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"

# City centers used in feature engineering (must match engineer_features.py!)
SAN_FRANCISCO = (37.77, -122.42)
LOS_ANGELES = (34.05, -118.24)


@st.cache_resource  # load the 140MB model only once, not on every slider move
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 California House Price Predictor")
st.caption("Random Forest trained on the California Housing dataset · R² = 0.805")

model = load_model()

# ---------------- Sidebar: user inputs ----------------
st.sidebar.header("District details")

med_inc = st.sidebar.slider(
    "Median income ($10,000s)", 0.5, 15.0, 3.7, 0.1,
    help="3.7 means $37,000/year"
)
house_age = st.sidebar.slider("Median house age (years)", 1, 52, 28)
ave_rooms = st.sidebar.slider("Average rooms per house", 1.0, 15.0, 5.2, 0.1)
ave_bedrms = st.sidebar.slider("Average bedrooms per house", 0.5, 4.0, 1.1, 0.1)
population = st.sidebar.slider("District population", 3, 10000, 1400)
ave_occup = st.sidebar.slider("Average people per household", 1.0, 10.0, 2.9, 0.1)

st.sidebar.header("Location")
city_preset = st.sidebar.selectbox(
    "Quick location preset",
    ["Custom", "San Francisco", "Los Angeles", "Sacramento", "Fresno (inland)"],
)
PRESETS = {
    "San Francisco": (37.77, -122.42),
    "Los Angeles": (34.05, -118.24),
    "Sacramento": (38.58, -121.49),
    "Fresno (inland)": (36.75, -119.77),
}
if city_preset != "Custom":
    preset_lat, preset_lon = PRESETS[city_preset]
else:
    preset_lat, preset_lon = 35.63, -119.56  # dataset average

latitude = st.sidebar.slider("Latitude", 32.5, 42.0, preset_lat, 0.01)
longitude = st.sidebar.slider("Longitude", -124.4, -114.3, preset_lon, 0.01)

# ---------------- Build the SAME features as training ----------------
rooms_per_person = ave_rooms / ave_occup
bedrms_ratio = ave_bedrms / ave_rooms
dist_to_sf = np.sqrt((latitude - SAN_FRANCISCO[0]) ** 2 + (longitude - SAN_FRANCISCO[1]) ** 2)
dist_to_la = np.sqrt((latitude - LOS_ANGELES[0]) ** 2 + (longitude - LOS_ANGELES[1]) ** 2)
dist_to_city = min(dist_to_sf, dist_to_la)

# Column order MUST match the training data exactly
input_df = pd.DataFrame([{
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude,
    "RoomsPerPerson": rooms_per_person,
    "BedrmsRatio": bedrms_ratio,
    "DistToSF": dist_to_sf,
    "DistToLA": dist_to_la,
    "DistToCity": dist_to_city,
}])

# ---------------- Predict ----------------
prediction = model.predict(input_df)[0]
price_dollars = prediction * 100_000

st.subheader("Predicted median house value")
st.metric(label="for this district", value=f"${price_dollars:,.0f}")
st.caption("Typical error: ~$29,000 · Prices reflect the 1990 census data the model was trained on")

# Show the district on a map
st.subheader("District location")
st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5)

# Show the inputs the model received (educational!)
with st.expander("See what the model received (13 features)"):
    st.dataframe(input_df.T.rename(columns={0: "value"}))
    st.caption(
        "RoomsPerPerson, BedrmsRatio and the 3 distances are computed "
        "automatically - the same feature engineering used in training."
    )
