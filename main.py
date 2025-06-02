from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sqlite3
import os

# Initialize FastAPI app
app = FastAPI()

# Set up database
DB_FILE = "crop_predictions.db"
def setup_database():
    # Check if database exists
    db_exists = os.path.isfile(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    if not db_exists:
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

# Call setup
setup_database()

# Load the trained LightGBM model
model = joblib.load("lightgbm_crop_yield_model.pkl")

# Load label encoders
label_encoders = joblib.load("label_encoders.pkl")

# Define request model
class CropInput(BaseModel):
    region: str
    soil_type: str
    crop: str
    avg_temp: float
    avg_rainfall: float
    weather: str
    days_to_harvest: int
    fertilizer_used: bool
    irrigation_used: bool

# Define mappings
region_mapping = {3: "West", 2: "South", 1: "North", 0: "East"}
soil_mapping = {4: "Sandy", 1: "Clay", 2: "Loam", 5: "Silt", 3: "Peaty", 0: "Chalky"}
crop_mapping = {1: "Cotton", 3: "Rice", 0: "Barley", 4: "Soybean", 5: "Wheat", 2: "Maize"}
weather_mapping = {0: "Cloudy", 1: "Rainy", 2: "Sunny"}

# Reverse mappings for API input
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
    return {"message": "Crop Yield Prediction API is running!"}

@app.post("/predict")
def predict_yield(data: CropInput):
    try:
        # Convert categorical inputs into encoded values
        region_encoded = region_options[data.region]
        soil_encoded = soil_options[data.soil_type]
        crop_encoded = crop_options[data.crop]
        weather_encoded = weather_options[data.weather]

        # Convert boolean values to 0/1
        fertilizer_value = int(data.fertilizer_used)
        irrigation_value = int(data.irrigation_used)

        # Prepare input data
        input_data = np.array([[region_encoded, soil_encoded, crop_encoded, data.avg_rainfall, data.avg_temp,
                                fertilizer_value, irrigation_value, weather_encoded, data.days_to_harvest]])

        # Predict using the ML model
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)  # Round for better readability

        # Get average yield for the given crop (default to 4.0 if not found)
        average_yield = crop_avg_yield.get(data.crop, 4.0)
        optimal_yield = round(average_yield * 1.2, 2)  # 20% above average

        # Determine recommendation based on yield
        recommendation = ""
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
                    "Increase nitrogen-based fertilizers (e.g., urea) for better crop growth.",
                    "Adjust irrigation timing based on soil moisture levels.",
                    "Ensure balanced crop rotation to maintain soil fertility."
                ]
            }
        else:
            recommendation = {
                "status": "High yield expected",
                "advice": [
                    "Maintain current farming techniques.",
                    "Keep using organic or chemical fertilizers as needed.",
                    "Monitor seasonal variations to sustain high yield production."
                ]
            }

        # Save prediction to database
        current_time = datetime.now().isoformat()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (date, crop, region, yield_value, created_at) VALUES (?, ?, ?, ?, ?)",
            (current_time, data.crop, data.region, prediction, current_time)
        )
        conn.commit()
        conn.close()
        
        return {
            "predicted_yield": prediction,
            "crop": data.crop,
            "region": data.region,
            "average_yield": average_yield,
            "optimal_yield": optimal_yield,
            "recommendation": recommendation
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/history")
def get_prediction_history(time_period: str = "all"):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    current_date = datetime.now()
    
    # Define time filters based on the requested period
    if time_period == "week":
        # Get records from the last 7 days
        start_date = (current_date.replace(hour=0, minute=0, second=0, microsecond=0) - 
                      datetime.timedelta(days=7)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    elif time_period == "month":
        # Get records from the last 30 days
        start_date = (current_date.replace(hour=0, minute=0, second=0, microsecond=0) - 
                      datetime.timedelta(days=30)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    elif time_period == "year":
        # Get records from the last 365 days
        start_date = (current_date.replace(hour=0, minute=0, second=0, microsecond=0) - 
                      datetime.timedelta(days=365)).isoformat()
        cursor.execute("SELECT date, crop, region, yield_value FROM predictions WHERE date >= ?", (start_date,))
    else:
        # Get all records
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
    
    # Get all current predictions
    cursor.execute("SELECT date, crop, region, yield_value FROM predictions")
    predictions = cursor.fetchall()
    
    # Insert into archived_predictions
    archive_time = datetime.now().isoformat()
    for pred in predictions:
        cursor.execute(
            "INSERT INTO archived_predictions (date, crop, region, yield_value, archived_at) VALUES (?, ?, ?, ?, ?)",
            (pred[0], pred[1], pred[2], pred[3], archive_time)
        )
    
    # Clear the predictions table
    cursor.execute("DELETE FROM predictions")
    
    conn.commit()
    conn.close()
    
    return {"message": f"Successfully archived {len(predictions)} predictions"}

@app.get("/history/stats")
def get_history_stats():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get count of predictions by crop type
    cursor.execute("SELECT crop, COUNT(*) FROM predictions GROUP BY crop")
    crop_counts = dict(cursor.fetchall())
    
    # Get count of predictions by region
    cursor.execute("SELECT region, COUNT(*) FROM predictions GROUP BY region")
    region_counts = dict(cursor.fetchall())
    
    # Get average yield by crop
    cursor.execute("SELECT crop, AVG(yield_value) FROM predictions GROUP BY crop")
    crop_avgs = {crop: round(avg, 2) for crop, avg in cursor.fetchall()}
    
    # Get count of total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_count = cursor.fetchone()[0]
    
    # Get count of total archived predictions
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific URLs in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Run the API
if __name__ != "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)