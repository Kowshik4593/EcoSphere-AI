from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import geopandas as gpd
from shapely.geometry import Point
import requests
import os

# === Load ML model ===
model = joblib.load('climate_safe_housing_model.pkl')

# === Load safe zones data ===
safe_data = pd.read_csv('safe_zones.csv')

# === Load ocean polygons (Natural Earth shapefile) ===
oceans = gpd.read_file("ne_10m_ocean.shp")

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request models ===
class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    flood_weight: float = 0.4
    heat_weight: float = 0.3
    drought_weight: float = 0.3

class ChatRequest(BaseModel):
    message: str

# === Haversine formula ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# === Ocean detection ===
def is_point_in_ocean(lat, lon):
    point = Point(lon, lat)
    return oceans.contains(point).any()

# === Climate risk generator ===
def get_risks_for_location(lat, lon):
    flood = round(((abs(lat) * 0.73 + abs(lon) * 0.21) % 1), 2)
    heat = round(((abs(lat) * 0.31 + abs(lon) * 0.47) % 1), 2)
    drought = round(((abs(lat) * 0.13 + abs(lon) * 0.59) % 1), 2)
    return flood, heat, drought

# === Main prediction route ===
@app.post("/predict")
def predict(location: LocationRequest):
    if is_point_in_ocean(location.latitude, location.longitude):
        return {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "housing_recommendation": "Location is in ocean/waterbody! 🌊",
            "nearest_safe_locations": [],
            "ocean": True,
            "confidence": 0.0,
            "flood_risk": None,
            "heat_risk": None,
            "drought_risk": None,
            "cri": None,
            "flood_weight": location.flood_weight,
            "heat_weight": location.heat_weight,
            "drought_weight": location.drought_weight
        }

    flood_val, heat_val, drought_val = get_risks_for_location(location.latitude, location.longitude)

    cri = (
        location.flood_weight * flood_val +
        location.heat_weight * heat_val +
        location.drought_weight * drought_val
    )

    input_data = pd.DataFrame({
        'latitude': [location.latitude],
        'longitude': [location.longitude],
        'occurrence': [cri]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Safe for Housing ✅"
        nearby_safe = []
    else:
        result = "Risky Zone ❌"
        safe_data['distance_km'] = safe_data.apply(lambda row: haversine(
            location.latitude, location.longitude, row['latitude'], row['longitude']), axis=1)
        nearest = safe_data.sort_values('distance_km').head(3)
        nearby_safe = nearest[['latitude', 'longitude', 'distance_km']].to_dict(orient='records')

    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "flood_weight": location.flood_weight,
        "heat_weight": location.heat_weight,
        "drought_weight": location.drought_weight,
        "flood_risk": flood_val,
        "heat_risk": heat_val,
        "drought_risk": drought_val,
        "cri": cri,
        "housing_recommendation": result,
        "nearest_safe_locations": nearby_safe,
        "ocean": False,
        "confidence": 1.0 if prediction == 1 else 0.85
    }


# === Vertex AI Chat Integration (Gemini, Explainable AI) ===
from google.cloud import aiplatform
from google.oauth2 import service_account

# Set up Vertex AI client (ensure service-account.json is present)
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
SERVICE_ACCOUNT_FILE = "service-account.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE
)
aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

def chat_with_vertex_ai(message: str):
    # Gemini 1.5 Pro is suitable for explainable AI and chat
    model = aiplatform.LanguageModel.from_pretrained("gemini-1.5-pro-preview-0409")
    system_instruction = "You are a helpful, explainable AI assistant for climate safety. Always explain your reasoning."
    response = model.predict(
        [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": message}
        ],
        temperature=0.7
    )
    # Vertex AI returns a list of candidates, take the first
    return response.candidates[0].content if response.candidates else "No response."


# === Chat endpoint using Vertex AI ===
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        reply = chat_with_vertex_ai(req.message)
        return {"reply": reply}
    except Exception as e:
        print(f"Vertex AI error: {e}")
        return {"reply": "Sorry, something went wrong while generating a response."}
