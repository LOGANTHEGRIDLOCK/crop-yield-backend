from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import sqlite3
import os
import requests
import asyncio
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Weather API configuration
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "82e612fbadcbde73be0b69fe6e2d4dca")  # Get from environment variable
WEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Nigerian regional coordinates (approximate centers)
NIGERIAN_REGIONS = {
    "North": {"lat": 12.0, "lon": 8.0, "cities": ["Kano", "Kaduna", "Maiduguri"]},
    "South": {"lat": 5.0, "lon": 7.0, "cities": ["Lagos", "Port Harcourt", "Calabar"]},
    "East": {"lat": 6.5, "lon": 7.5, "cities": ["Enugu", "Onitsha", "Aba"]},
    "West": {"lat": 7.5, "lon": 3.5, "cities": ["Ibadan", "Abeokuta", "Ilorin"]}
}

# Set up database
DB_FILE = "crop_predictions.db"
def setup_database():
    db_exists = os.path.isfile(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    if not db_exists:
        cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            crop TEXT,
            region TEXT,
            yield_value REAL,
            weather_data TEXT,
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
            weather_data TEXT,
            archived_at TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE weather_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            weather_data TEXT,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            weather_condition TEXT,
            cached_at TEXT,
            UNIQUE(region)
        )
        ''')
        
        conn.commit()
    
    conn.close()

# Call setup
setup_database()

# Load the trained LightGBM model
try:
    model = joblib.load("lightgbm_crop_yield_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except FileNotFoundError:
    logger.warning("Model files not found. Make sure to train and save your model first.")
    model = None
    label_encoders = None

# Define request models
class CropInput(BaseModel):
    region: str
    soil_type: str
    crop: str
    days_to_harvest: int
    fertilizer_used: bool
    irrigation_used: bool
    use_weather_api: bool = True  # New field to toggle weather API usage

class ManualWeatherInput(BaseModel):
    region: str
    soil_type: str
    crop: str
    avg_temp: float
    avg_rainfall: float
    weather: str
    days_to_harvest: int
    fertilizer_used: bool
    irrigation_used: bool

# Weather data fetching functions
async def fetch_weather_data(region: str) -> dict:
    """Fetch weather data from OpenWeatherMap API for a Nigerian region"""
    try:
        if region not in NIGERIAN_REGIONS:
            raise ValueError(f"Region {region} not supported. Available regions: {list(NIGERIAN_REGIONS.keys())}")
        
        coords = NIGERIAN_REGIONS[region]
        
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": WEATHER_API_KEY,
            "units": "metric"  # Celsius
        }
        
        response = requests.get(WEATHER_BASE_URL, params=params, timeout=10)
        
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid weather API key")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Weather API request failed")
        
        data = response.json()
        
        # Extract relevant weather information
        weather_info = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "weather_condition": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "clouds": data["clouds"]["all"],
            "region": region,
            "city": data.get("name", f"{region} Region"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Estimate rainfall based on weather conditions and humidity
        rainfall = estimate_rainfall(weather_info)
        weather_info["estimated_rainfall"] = rainfall
        
        # Cache the weather data
        cache_weather_data(region, weather_info)
        
        return weather_info
        
    except requests.exceptions.Timeout:
        logger.error(f"Weather API timeout for region {region}")
        return get_cached_weather_data(region)
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed for region {region}: {e}")
        return get_cached_weather_data(region)
    except Exception as e:
        logger.error(f"Error fetching weather data for region {region}: {e}")
        return get_cached_weather_data(region)

def estimate_rainfall(weather_info: dict) -> float:
    """Estimate rainfall based on weather conditions, humidity, and clouds"""
    humidity = weather_info["humidity"]
    clouds = weather_info["clouds"]
    weather_condition = weather_info["weather_condition"].lower()
    
    # Base rainfall estimation
    if "rain" in weather_condition:
        base_rainfall = 15.0  # mm
    elif "drizzle" in weather_condition:
        base_rainfall = 5.0
    elif "thunderstorm" in weather_condition:
        base_rainfall = 25.0
    elif "snow" in weather_condition:
        base_rainfall = 8.0  # Snow water equivalent
    else:
        base_rainfall = 0.0
    
    # Adjust based on humidity and cloud cover
    humidity_factor = min(humidity / 100.0, 1.0)
    cloud_factor = min(clouds / 100.0, 1.0)
    
    # If no precipitation but high humidity and clouds, estimate light rain
    if base_rainfall == 0 and humidity > 80 and clouds > 70:
        base_rainfall = 2.0
    
    estimated_rainfall = base_rainfall * (0.5 + 0.5 * humidity_factor) * (0.7 + 0.3 * cloud_factor)
    
    return round(estimated_rainfall, 2)

def cache_weather_data(region: str, weather_info: dict):
    """Cache weather data in database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Insert or replace weather data
        cursor.execute('''
            INSERT OR REPLACE INTO weather_cache 
            (region, weather_data, temperature, humidity, rainfall, weather_condition, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            region,
            str(weather_info),
            weather_info["temperature"],
            weather_info["humidity"],
            weather_info["estimated_rainfall"],
            weather_info["weather_condition"],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error caching weather data: {e}")

def get_cached_weather_data(region: str) -> dict:
    """Get cached weather data if API fails"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT weather_data, temperature, humidity, rainfall, weather_condition, cached_at
            FROM weather_cache WHERE region = ?
        ''', (region,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "temperature": result[1],
                "humidity": result[2],
                "estimated_rainfall": result[3],
                "weather_condition": result[4],
                "cached_at": result[5],
                "is_cached": True,
                "region": region
            }
        else:
            # Return default values if no cached data
            return get_default_weather_data(region)
            
    except Exception as e:
        logger.error(f"Error getting cached weather data: {e}")
        return get_default_weather_data(region)

def get_default_weather_data(region: str) -> dict:
    """Return default weather data based on Nigerian regional patterns"""
    defaults = {
        "North": {"temp": 30.0, "rainfall": 2.0, "weather": "Sunny", "humidity": 45},
        "South": {"temp": 27.0, "rainfall": 8.0, "weather": "Cloudy", "humidity": 75},
        "East": {"temp": 28.0, "rainfall": 6.0, "weather": "Cloudy", "humidity": 65},
        "West": {"temp": 29.0, "rainfall": 5.0, "weather": "Sunny", "humidity": 60}
    }
    
    default = defaults.get(region, defaults["South"])
    
    return {
        "temperature": default["temp"],
        "estimated_rainfall": default["rainfall"],
        "weather_condition": default["weather"],
        "humidity": default["humidity"],
        "is_default": True,
        "region": region
    }

# Define mappings (same as before)
region_mapping = {3: "West", 2: "South", 1: "North", 0: "East"}
soil_mapping = {4: "Sandy", 1: "Clay", 2: "Loam", 5: "Silt", 3: "Peaty", 0: "Chalky"}
crop_mapping = {1: "Cotton", 3: "Rice", 0: "Barley", 4: "Soybean", 5: "Wheat", 2: "Maize"}
weather_mapping = {0: "Cloudy", 1: "Rainy", 2: "Sunny"}

# Reverse mappings
region_options = {v: k for k, v in region_mapping.items()}
soil_options = {v: k for k, v in soil_mapping.items()}
crop_options = {v: k for k, v in crop_mapping.items()}
weather_options = {v: k for k, v in weather_mapping.items()}

# Average yield for different crops
crop_avg_yield = {
    "Soybean": 3.2, "Wheat": 3.5, "Maize": 5.8,
    "Rice": 4.6, "Cotton": 2.1, "Barley": 3.9
}

@app.get("/")
def home():
    return {"message": "Enhanced Crop Yield Prediction API with Weather Integration is running!"}

@app.get("/weather/{region}")
async def get_weather(region: str):
    """Get current weather data for a Nigerian region"""
    try:
        weather_data = await fetch_weather_data(region)
        return {"weather": weather_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/regions")
def get_regions():
    """Get available Nigerian regions"""
    return {
        "regions": list(NIGERIAN_REGIONS.keys()),
        "details": NIGERIAN_REGIONS
    }

@app.post("/predict")
async def predict_yield(data: CropInput):
    """Predict crop yield with automatic weather data fetching"""
    if not model or not label_encoders:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure model files are available.")
    
    try:
        # Fetch weather data if requested
        if data.use_weather_api:
            weather_info = await fetch_weather_data(data.region)
            avg_temp = weather_info["temperature"]
            avg_rainfall = weather_info["estimated_rainfall"]
            
            # Map weather condition to our model's expected format
            weather_condition = weather_info["weather_condition"]
            if weather_condition.lower() in ["rain", "drizzle", "thunderstorm"]:
                weather = "Rainy"
            elif weather_condition.lower() in ["clear", "sun"]:
                weather = "Sunny"
            else:
                weather = "Cloudy"
        else:
            # Use default values or raise error
            raise HTTPException(status_code=400, detail="Manual weather input not provided. Use /predict-manual endpoint for manual input.")

        # Convert categorical inputs
        region_encoded = region_options[data.region]
        soil_encoded = soil_options[data.soil_type]
        crop_encoded = crop_options[data.crop]
        weather_encoded = weather_options[weather]

        # Convert boolean values
        fertilizer_value = int(data.fertilizer_used)
        irrigation_value = int(data.irrigation_used)

        # Prepare input data
        input_data = np.array([[region_encoded, soil_encoded, crop_encoded, avg_rainfall, avg_temp,
                                fertilizer_value, irrigation_value, weather_encoded, data.days_to_harvest]])

        # Predict using the ML model
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        # Get average yield for the given crop
        average_yield = crop_avg_yield.get(data.crop, 4.0)
        optimal_yield = round(average_yield * 1.2, 2)

        # Generate recommendations
        recommendation = generate_recommendations(prediction, weather_info, data)

        # Save prediction to database
        current_time = datetime.now().isoformat()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (date, crop, region, yield_value, weather_data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (current_time, data.crop, data.region, prediction, str(weather_info), current_time)
        )
        conn.commit()
        conn.close()
        
        return {
            "predicted_yield": prediction,
            "crop": data.crop,
            "region": data.region,
            "average_yield": average_yield,
            "optimal_yield": optimal_yield,
            "weather_data": weather_info,
            "recommendation": recommendation
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-manual")
def predict_yield_manual(data: ManualWeatherInput):
    """Predict crop yield with manual weather input"""
    if not model or not label_encoders:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Convert categorical inputs
        region_encoded = region_options[data.region]
        soil_encoded = soil_options[data.soil_type]
        crop_encoded = crop_options[data.crop]
        weather_encoded = weather_options[data.weather]

        # Convert boolean values
        fertilizer_value = int(data.fertilizer_used)
        irrigation_value = int(data.irrigation_used)

        # Prepare input data
        input_data = np.array([[region_encoded, soil_encoded, crop_encoded, data.avg_rainfall, data.avg_temp,
                                fertilizer_value, irrigation_value, weather_encoded, data.days_to_harvest]])

        # Predict using the ML model
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        # Get average yield
        average_yield = crop_avg_yield.get(data.crop, 4.0)
        optimal_yield = round(average_yield * 1.2, 2)

        # Generate basic recommendations
        recommendation = generate_basic_recommendations(prediction)

        # Save prediction to database
        current_time = datetime.now().isoformat()
        weather_info = {
            "temperature": data.avg_temp,
            "estimated_rainfall": data.avg_rainfall,
            "weather_condition": data.weather,
            "manual_input": True
        }
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (date, crop, region, yield_value, weather_data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (current_time, data.crop, data.region, prediction, str(weather_info), current_time)
        )
        conn.commit()
        conn.close()
        
        return {
            "predicted_yield": prediction,
            "crop": data.crop,
            "region": data.region,
            "average_yield": average_yield,
            "optimal_yield": optimal_yield,
            "weather_data": weather_info,
            "recommendation": recommendation
        }

    except Exception as e:
        logger.error(f"Manual prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_recommendations(prediction: float, weather_info: dict, data: CropInput) -> dict:
    """Generate enhanced recommendations based on weather data"""
    temp = weather_info.get("temperature", 25)
    rainfall = weather_info.get("estimated_rainfall", 5)
    humidity = weather_info.get("humidity", 60)
    
    advice = []
    
    # Base yield recommendations
    if prediction < 3:
        status = "Low yield detected"
        advice.extend([
            "Use organic compost to enrich soil nutrients.",
            "Improve irrigation by ensuring crops get enough water.",
            "Monitor soil pH and adjust with lime or sulfur if needed."
        ])
    elif prediction < 6:
        status = "Moderate yield detected"
        advice.extend([
            "Increase nitrogen-based fertilizers (e.g., urea) for better crop growth.",
            "Adjust irrigation timing based on soil moisture levels.",
            "Ensure balanced crop rotation to maintain soil fertility."
        ])
    else:
        status = "High yield expected"
        advice.extend([
            "Maintain current farming techniques.",
            "Keep using organic or chemical fertilizers as needed.",
            "Monitor seasonal variations to sustain high yield production."
        ])
    
    # Weather-based recommendations
    if temp > 35:
        advice.append("High temperature detected. Consider shade nets or increased irrigation.")
    elif temp < 15:
        advice.append("Low temperature may affect growth. Consider protective measures.")
    
    if rainfall > 20:
        advice.append("Heavy rainfall expected. Ensure proper drainage to prevent waterlogging.")
    elif rainfall < 2:
        advice.append("Low rainfall. Increase irrigation frequency and consider mulching.")
    
    if humidity > 80:
        advice.append("High humidity may increase disease risk. Monitor for fungal infections.")
    elif humidity < 40:
        advice.append("Low humidity detected. Consider moisture retention techniques.")
    
    return {
        "status": status,
        "advice": advice,
        "weather_advisory": f"Current conditions: {temp}°C, {rainfall}mm rainfall"
    }

def generate_basic_recommendations(prediction: float) -> dict:
    """Generate basic recommendations without weather data"""
    if prediction < 3:
        return {
            "status": "Low yield detected",
            "advice": [
                "Use organic compost to enrich soil nutrients.",
                "Improve irrigation by ensuring crops get enough water.",
                "Monitor soil pH and adjust with lime or sulfur if needed."
            ]
        }
    elif prediction < 6:
        return {
            "status": "Moderate yield detected",
            "advice": [
                "Increase nitrogen-based fertilizers (e.g., urea) for better crop growth.",
                "Adjust irrigation timing based on soil moisture levels.",
                "Ensure balanced crop rotation to maintain soil fertility."
            ]
        }
    else:
        return {
            "status": "High yield expected",
            "advice": [
                "Maintain current farming techniques.",
                "Keep using organic or chemical fertilizers as needed.",
                "Monitor seasonal variations to sustain high yield production."
            ]
        }

# Keep existing endpoints (history, archive, stats) unchanged
@app.get("/history")
def get_prediction_history(time_period: str = "all"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    current_date = datetime.now()
    
    if time_period == "week":
        start_date = (current_date.replace(hour=0, minute=0, second=0, microsecond=0) - 
                      timedelta(days=7)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    elif time_period == "month":
        start_date = (current_date.replace(hour=0, minute=0, second=0, microsecond=0) - 
                      timedelta(days=30)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    elif time_period == "year":
        start_date = (current_date.replace(hour=0, minute=0, second=0, microsecond=0) - 
                      timedelta(days=365)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    else:
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions")
    
    rows = cursor.fetchall()
    history = []
    for row in rows:
        history.append({
            "date": row[0],
            "crop": row[1],
            "region": row[2],
            "yield": row[3]
        })
    
    conn.close()
    return {"history": history}

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
    
    return {"message": f"Successfully archived {len(predictions)} predictions"}

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)