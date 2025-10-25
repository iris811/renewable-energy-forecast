#!/bin/bash
# Start FastAPI server

echo "Starting Renewable Energy Forecasting API..."
echo "API will be available at http://localhost:8000"
echo "API docs will be available at http://localhost:8000/docs"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
