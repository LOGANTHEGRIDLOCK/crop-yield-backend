from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

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

# Store prediction history
prediction_history = []

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

        # Save to history
        new_prediction = {
            "date": datetime.now().isoformat(),
            "crop": data.crop,
            "region": data.region,
            "yield": prediction
        }
        prediction_history.append(new_prediction)
        
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
def get_prediction_history():
    return {"history": prediction_history}

archived_predictions = []

@app.post("/archive")
def archive_predictions():
    global prediction_history, archived_predictions
    archived_predictions.extend(prediction_history)
    prediction_history.clear()
    return {"message": "Predictions archived successfully"}


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
