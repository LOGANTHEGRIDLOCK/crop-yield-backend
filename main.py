from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from supabase import create_client, Client
import requests

import os

# Load environment variables


# Initialize FastAPI app
app = FastAPI()

# Initialize Supabase client
SUPABASE_URL = "https://gffzsgzzhqfyhglounko.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdmZnpzZ3p6aHFmeWhnbG91bmtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg4ODAyOTIsImV4cCI6MjA2NDQ1NjI5Mn0.Kcip4FIem4R6w42dHzibEY6GV5bmnYF7joZZWoVja8w"
OPENWEATHER_API_KEY ="82e612fbadcbde73be0b69fe6e2d4dca"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in your .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load the trained LightGBM model
try:
    model = joblib.load("lightgbm_crop_yield_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    model = None
    label_encoders = None

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

# Models
class CropInput(BaseModel):
    soil_type: str
    crop: str
    days_to_harvest: int
    fertilizer_used: bool
    irrigation_used: bool
    latitude: float
    longitude: float

# Utilities check
def get_weather_data(lat: float, lon: float):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Weather API error: {response.text}")

    data = response.json()
    avg_temp = data["main"]["temp"]
    rainfall = data.get("rain", {}).get("1h", 0.0)
    weather_main = data["weather"][0]["main"]

    if weather_main.lower() in ["clear"]:
        weather = "Sunny"
    elif weather_main.lower() in ["clouds"]:
        weather = "Cloudy"
    else:
        weather = "Rainy"

    return avg_temp, rainfall, weather

def get_region_from_latlon(lat: float, lon: float) -> str:
    if lat >= 10.0:
        return "North"
    elif lat >= 6.0 and lon <= 5.0:
        return "West"
    elif lat >= 6.0 and lon >= 8.0:
        return "East"
    else:
        return "South"


@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running with Supabase!"}

@app.get("/health")
def health_check():
    """Check if the API and database connection are working"""
    try:
        # Test Supabase connection
        response = supabase.table("predictions").select("count", count="exact").execute()
        return {
            "status": "healthy",
            "database": "connected",
            "model_loaded": model is not None,
            "total_predictions": response.count if response.count else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


@app.post("/predict")
def predict_yield(data: CropInput):
    if model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        avg_temp, avg_rainfall, weather_str = get_weather_data(data.latitude, data.longitude)
        region_str = get_region_from_latlon(data.latitude, data.longitude)

        if data.soil_type not in soil_options:
            raise HTTPException(status_code=400, detail=f"Invalid soil type. Options: {list(soil_options.keys())}")
        if data.crop not in crop_options:
            raise HTTPException(status_code=400, detail=f"Invalid crop. Options: {list(crop_options.keys())}")
        if weather_str not in weather_options:
            raise HTTPException(status_code=400, detail=f"Unrecognized weather condition: {weather_str}")
        if region_str not in region_options:
            raise HTTPException(status_code=400, detail=f"Invalid region: {region_str}")

        region_encoded = region_options[region_str]
        soil_encoded = soil_options[data.soil_type]
        crop_encoded = crop_options[data.crop]
        weather_encoded = weather_options[weather_str]
        fertilizer_value = int(data.fertilizer_used)
        irrigation_value = int(data.irrigation_used)

        input_data = np.array([[region_encoded, soil_encoded, crop_encoded, avg_rainfall, avg_temp,
                                fertilizer_value, irrigation_value, weather_encoded, data.days_to_harvest]])

        prediction = round(model.predict(input_data)[0], 2)
        average_yield = crop_avg_yield.get(data.crop, 4.0)
        optimal_yield = round(average_yield * 1.2, 2)

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

        now = datetime.now().isoformat()
        prediction_data = {
            "date": now,
            "crop": data.crop,
            "region": region_str,
            "yield_value": prediction,
            "created_at": now
        }

        result = supabase.table("predictions").insert(prediction_data).execute()

        return {
            "predicted_yield": prediction,
            "crop": data.crop,
            "region": region_str,
            "average_yield": average_yield,
            "optimal_yield": optimal_yield,
            "recommendation": recommendation,
            "prediction_id": result.data[0]["id"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.get("/history")
def get_prediction_history(time_period: str = "all"):
    try:
        current_date = datetime.now()
        
        # Build query based on time period
        query = supabase.table("predictions").select("date,crop,region,yield_value,id")
        
        if time_period == "week":
            start_date = (current_date - timedelta(days=7)).isoformat()
            query = query.gte("date", start_date)
        elif time_period == "month":
            start_date = (current_date - timedelta(days=30)).isoformat()
            query = query.gte("date", start_date)
        elif time_period == "year":
            start_date = (current_date - timedelta(days=365)).isoformat()
            query = query.gte("date", start_date)
        
        # Execute query with ordering
        result = query.order("created_at", desc=True).execute()
        
        history = []
        for row in result.data:
            history.append({
                "id": row["id"],
                "date": row["date"],
                "crop": row["crop"],
                "region": row["region"],
                "yield": row["yield_value"]
            })
        
        return {
            "history": history,
            "count": len(history),
            "time_period": time_period
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.post("/archive")
def archive_predictions():
    try:
        # Get all current predictions
        predictions_result = supabase.table("predictions").select("*").execute()
        predictions = predictions_result.data
        
        if not predictions:
            return {"message": "No predictions to archive"}
        
        # Prepare data for archiving
        archive_time = datetime.now().isoformat()
        archive_data = []
        
        for pred in predictions:
            archive_data.append({
                "date": pred["date"],
                "crop": pred["crop"],
                "region": pred["region"],
                "yield_value": pred["yield_value"],
                "archived_at": archive_time
            })
        
        # Insert into archived_predictions
        archive_result = supabase.table("archived_predictions").insert(archive_data).execute()
        
        if not archive_result.data:
            raise HTTPException(status_code=500, detail="Failed to archive predictions")
        
        # Clear the predictions table
        delete_result = supabase.table("predictions").delete().neq("id", 0).execute()
        
        return {
            "message": f"Successfully archived {len(predictions)} predictions",
            "archived_count": len(archive_result.data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive operation failed: {str(e)}")

@app.get("/history/stats")
def get_history_stats():
    try:
        # Get all predictions for analysis
        all_predictions = supabase.table("predictions").select("*").execute()
        predictions = all_predictions.data
        
        # Get archived predictions count
        archived_result = supabase.table("archived_predictions").select("id", count="exact").execute()
        total_archived = archived_result.count if archived_result.count else 0
        
        if not predictions:
            return {
                "total_predictions": 0,
                "total_archived": total_archived,
                "by_crop": {},
                "by_region": {},
                "average_yields": {}
            }
        
        # Calculate statistics
        crop_counts = {}
        region_counts = {}
        crop_yields = {}
        
        for pred in predictions:
            crop = pred["crop"]
            region = pred["region"]
            yield_val = pred["yield_value"]
            
            # Count by crop
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
            
            # Count by region
            region_counts[region] = region_counts.get(region, 0) + 1
            
            # Collect yields by crop for averaging
            if crop not in crop_yields:
                crop_yields[crop] = []
            crop_yields[crop].append(yield_val)
        
        # Calculate average yields
        crop_avgs = {}
        for crop, yields in crop_yields.items():
            crop_avgs[crop] = round(sum(yields) / len(yields), 2)
        
        return {
            "total_predictions": len(predictions),
            "total_archived": total_archived,
            "by_crop": crop_counts,
            "by_region": region_counts,
            "average_yields": crop_avgs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: int):
    """Delete a specific prediction by ID"""
    try:
        result = supabase.table("predictions").delete().eq("id", prediction_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {"message": f"Prediction {prediction_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction: {str(e)}")

@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    """Get a specific prediction by ID"""
    try:
        result = supabase.table("predictions").select("*").eq("id", prediction_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {"prediction": result.data[0]}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction: {str(e)}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific URLs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)