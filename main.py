from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import requests
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import sqlite3
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_FILE = "crop_predictions.db"
def setup_database():
    if not os.path.isfile(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                crop TEXT,
                region TEXT,
                yield_value REAL,
                created_at TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE archived_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                crop TEXT,
                region TEXT,
                yield_value REAL,
                archived_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

setup_database()

# Load trained model and encoders
model = joblib.load("lightgbm_crop_yield_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Weather API key (set your real key here)
WEATHER_API_KEY = "82e612fbadcbde73be0b69fe6e2d4dca"

def get_weather_data(region: str):
    # Map Nigerian regions to major cities
    region_city_map = {
        "North": "Kano",
        "South": "Lagos",
        "East": "Enugu",
        "West": "Ibadan"
    }
    city = region_city_map.get(region, "Abuja")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},NG&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        rainfall = data.get("rain", {}).get("1h", 0.0)
        weather_main = data["weather"][0]["main"]

        if "rain" in weather_main.lower():
            weather = "Rainy"
        elif "cloud" in weather_main.lower():
            weather = "Cloudy"
        else:
            weather = "Sunny"

        return temp, rainfall, weather
    else:
        raise Exception(f"Weather data fetch failed for region '{region}'")

# Define mappings
region_options = {"North": 0, "South": 1, "East": 2, "West": 3}
soil_options = {"Sandy": 4, "Clay": 1, "Loam": 2, "Silt": 5, "Peaty": 3, "Chalky": 0}
crop_options = {"Cotton": 1, "Rice": 3, "Barley": 0, "Soybean": 4, "Wheat": 5, "Maize": 2}
weather_options = {"Cloudy": 0, "Rainy": 1, "Sunny": 2}

crop_avg_yield = {
    "Soybean": 3.2, "Wheat": 3.5, "Maize": 5.8,
    "Rice": 4.6, "Cotton": 2.1, "Barley": 3.9
}

# Input schema
class CropInput(BaseModel):
    region: str
    soil_type: str
    crop: str
    days_to_harvest: int
    fertilizer_used: bool
    irrigation_used: bool

@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running!"}

@app.post("/predict")
def predict_yield(data: CropInput):
    try:
        # Fetch weather using Nigerian region
        avg_temp, avg_rainfall, weather = get_weather_data(data.region)

        # Encode values
        region_encoded = region_options[data.region]
        soil_encoded = soil_options[data.soil_type]
        crop_encoded = crop_options[data.crop]
        weather_encoded = weather_options[weather]
        fertilizer_value = int(data.fertilizer_used)
        irrigation_value = int(data.irrigation_used)

        input_data = np.array([[region_encoded, soil_encoded, crop_encoded, avg_rainfall, avg_temp,
                                fertilizer_value, irrigation_value, weather_encoded, data.days_to_harvest]])

        prediction = round(model.predict(input_data)[0], 2)
        average_yield = crop_avg_yield.get(data.crop, 4.0)
        optimal_yield = round(average_yield * 1.2, 2)

        # Recommendation logic
        if prediction < 3:
            recommendation = {
                "status": "Low yield detected",
                "advice": [
                    "Use organic compost to enrich soil nutrients.",
                    "Improve irrigation by ensuring crops get enough water.",
                    "Monitor soil pH and adjust with lime or sulfur if needed."
                ]
            }
        elif prediction < 6:
            recommendation = {
                "status": "Moderate yield detected",
                "advice": [
                    "Increase nitrogen-based fertilizers (e.g., urea).",
                    "Adjust irrigation based on moisture levels.",
                    "Practice crop rotation to maintain fertility."
                ]
            }
        else:
            recommendation = {
                "status": "High yield expected",
                "advice": [
                    "Maintain current techniques.",
                    "Continue balanced fertilizer use.",
                    "Monitor seasonal patterns regularly."
                ]
            }

        # Save to DB
        current_time = datetime.now().isoformat()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (date, crop, region, yield_value, created_at) VALUES (?, ?, ?, ?, ?)",
                       (current_time, data.crop, data.region, prediction, current_time))
        conn.commit()
        conn.close()

        return {
            "predicted_yield": prediction,
            "crop": data.crop,
            "region": data.region,
            "average_yield": average_yield,
            "optimal_yield": optimal_yield,
            "recommendation": recommendation,
            "weather": weather,
            "avg_temp": avg_temp,
            "avg_rainfall": avg_rainfall
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_prediction_history(time_period: str = "all"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_date = datetime.now()

    if time_period == "week":
        start_date = (current_date - timedelta(days=7)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    elif time_period == "month":
        start_date = (current_date - timedelta(days=30)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    elif time_period == "year":
        start_date = (current_date - timedelta(days=365)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    else:
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions")

    rows = cursor.fetchall()
    conn.close()
    return {"history": [{"date": row[0], "crop": row[1], "region": row[2], "yield": row[3]} for row in rows]}

@app.post("/archive")
def archive_predictions():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT date, crop, region, yield_value FROM predictions")
    predictions = cursor.fetchall()

    archive_time = datetime.now().isoformat()
    for pred in predictions:
        cursor.execute(
            "INSERT INTO archived_predictions (date, crop, region, yield_value, archived_at) VALUES (?, ?, ?, ?, ?)",
            (pred[0], pred[1], pred[2], pred[3], archive_time)
        )

    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return {"message": f"Archived {len(predictions)} predictions"}

@app.get("/history/stats")
def get_history_stats():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT crop, COUNT(*) FROM predictions GROUP BY crop")
    crop_counts = dict(cursor.fetchall())

    cursor.execute("SELECT region, COUNT(*) FROM predictions GROUP BY region")
    region_counts = dict(cursor.fetchall())

    cursor.execute("SELECT crop, AVG(yield_value) FROM predictions GROUP BY crop")
    crop_avgs = {crop: round(avg, 2) for crop, avg in cursor.fetchall()}

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM archived_predictions")
    total_archived = cursor.fetchone()[0]

    conn.close()

    return {
        "total_predictions": total_count,
        "total_archived": total_archived,
        "by_crop": crop_counts,
        "by_region": region_counts,
        "average_yields": crop_avgs
    }
