"""
FastAPI server for Renewable Energy Forecasting
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import torch
import numpy as np
from pathlib import Path
import yaml

from src.models.model_utils import load_model
from src.preprocessing.scaler import load_scaler
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Renewable Energy Forecasting API",
    description="API for predicting renewable energy generation",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and config
MODEL_CACHE = {}
SCALER_CACHE = {}
CONFIG = None


# Pydantic models
class WeatherData(BaseModel):
    """Weather data for prediction"""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity in %")
    pressure: float = Field(..., description="Pressure in hPa")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction in degrees")
    ghi: Optional[float] = Field(None, description="Global Horizontal Irradiance")
    dni: Optional[float] = Field(None, description="Direct Normal Irradiance")
    dhi: Optional[float] = Field(None, description="Diffuse Horizontal Irradiance")
    cloud_cover: Optional[float] = Field(None, ge=0, le=100, description="Cloud cover in %")
    precipitation: Optional[float] = Field(0.0, ge=0, description="Precipitation in mm")


class PredictionRequest(BaseModel):
    """Prediction request"""
    location: str = Field(..., description="Location name")
    energy_type: str = Field(..., description="Energy type (solar/wind)")
    weather_sequence: List[WeatherData] = Field(..., description="Sequence of weather data")
    timestamp: Optional[str] = Field(None, description="Timestamp for prediction")


class PredictionResponse(BaseModel):
    """Prediction response"""
    location: str
    energy_type: str
    timestamp: str
    predictions: List[float]
    unit: str = "kW"
    horizon_hours: int
    model_version: str


class ModelInfo(BaseModel):
    """Model information"""
    location: str
    energy_type: str
    model_type: str
    parameters: int
    last_updated: str
    available: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    version: str


# Utility functions
def load_config():
    """Load configuration"""
    global CONFIG
    if CONFIG is None:
        config_path = Path('configs/config.yaml')
        with open(config_path, 'r') as f:
            CONFIG = yaml.safe_load(f)
    return CONFIG


def get_model_key(location: str, energy_type: str) -> str:
    """Generate cache key for model"""
    return f"{location}_{energy_type}"


def load_model_and_scaler(location: str, energy_type: str, model_path: Optional[str] = None):
    """
    Load model and scaler from cache or disk

    Args:
        location: Location name
        energy_type: Energy type
        model_path: Optional path to model checkpoint

    Returns:
        tuple: (model, feature_scaler, target_scaler)
    """
    key = get_model_key(location, energy_type)

    # Check cache
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    # Load from disk
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Determine model path
        if model_path is None:
            model_path = Path('models/checkpoints/best_model.pth')

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        model = load_model(str(model_path), device=device)
        logger.info(f"Loaded model for {location} - {energy_type}")

        # Load scalers
        scaler_dir = Path('models/scalers')
        feature_scaler_path = scaler_dir / f"{location}_{energy_type}_feature_scaler.pkl"
        target_scaler_path = scaler_dir / f"{location}_{energy_type}_target_scaler.pkl"

        feature_scaler = None
        target_scaler = None

        if feature_scaler_path.exists():
            feature_scaler = load_scaler(str(feature_scaler_path))
            logger.info(f"Loaded feature scaler")

        if target_scaler_path.exists():
            target_scaler = load_scaler(str(target_scaler_path))
            logger.info(f"Loaded target scaler")

        # Cache
        MODEL_CACHE[key] = (model, feature_scaler, target_scaler)

        return model, feature_scaler, target_scaler

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Renewable Energy Forecasting API...")
    load_config()
    logger.info("API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    MODEL_CACHE.clear()
    SCALER_CACHE.clear()


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Renewable Energy Forecasting API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(MODEL_CACHE),
        version="0.1.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict renewable energy generation

    Args:
        request: Prediction request with weather data

    Returns:
        Prediction response with forecasted generation
    """
    try:
        # Load model and scalers
        model, feature_scaler, target_scaler = load_model_and_scaler(
            request.location,
            request.energy_type
        )

        # Prepare input data
        weather_data = []
        for w in request.weather_sequence:
            weather_dict = w.dict()
            weather_data.append(weather_dict)

        # Convert to features (simplified - in production, use proper feature engineering)
        features = []
        for w in weather_data:
            feature_vector = [
                w['temperature'],
                w['humidity'],
                w['pressure'],
                w['wind_speed'],
                w['wind_direction'],
                w.get('ghi', 0),
                w.get('dni', 0),
                w.get('dhi', 0),
                w.get('cloud_cover', 0),
                w.get('precipitation', 0)
            ]
            features.append(feature_vector)

        features = np.array(features)

        # Scale features if scaler available
        if feature_scaler is not None:
            features = feature_scaler.transform(features)

        # Convert to tensor
        device = next(model.parameters()).device
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)

        # Convert to numpy
        prediction = prediction.cpu().numpy()[0]

        # Inverse transform if scaler available
        if target_scaler is not None:
            prediction = target_scaler.inverse_transform(prediction.reshape(1, -1))[0]

        # Ensure non-negative
        prediction = np.maximum(prediction, 0)

        # Prepare response
        config = load_config()
        timestamp = request.timestamp or datetime.now().isoformat()

        response = PredictionResponse(
            location=request.location,
            energy_type=request.energy_type,
            timestamp=timestamp,
            predictions=prediction.tolist(),
            horizon_hours=config['model']['prediction_horizon'],
            model_version="0.1.0"
        )

        logger.info(f"Prediction successful for {request.location} - {request.energy_type}")

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List available models

    Returns:
        List of available models
    """
    models = []

    checkpoints_dir = Path('models/checkpoints')
    if not checkpoints_dir.exists():
        return models

    # Scan for available models
    for model_path in checkpoints_dir.glob('*.pth'):
        try:
            # Load model to get info
            device = 'cpu'
            model = load_model(str(model_path), device=device)

            num_params = sum(p.numel() for p in model.parameters())

            model_info = ModelInfo(
                location="Unknown",  # Would need to parse from filename or metadata
                energy_type="Unknown",
                model_type="LSTM",  # Would need to detect
                parameters=num_params,
                last_updated=datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
                available=True
            )

            models.append(model_info)

        except Exception as e:
            logger.warning(f"Could not load model {model_path}: {e}")
            continue

    return models


@app.post("/models/load")
async def load_model_endpoint(
    location: str,
    energy_type: str,
    model_path: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Load model into cache

    Args:
        location: Location name
        energy_type: Energy type
        model_path: Optional path to model checkpoint
        background_tasks: Background tasks

    Returns:
        Success message
    """
    try:
        load_model_and_scaler(location, energy_type, model_path)

        return {
            "status": "success",
            "message": f"Model loaded for {location} - {energy_type}",
            "cached_models": len(MODEL_CACHE)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/cache")
async def clear_cache():
    """
    Clear model cache

    Returns:
        Success message
    """
    cleared = len(MODEL_CACHE)
    MODEL_CACHE.clear()
    SCALER_CACHE.clear()

    logger.info(f"Cleared {cleared} models from cache")

    return {
        "status": "success",
        "message": f"Cleared {cleared} models from cache"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
