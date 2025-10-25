@echo off
REM Start FastAPI server on Windows

echo Starting Renewable Energy Forecasting API...
echo API will be available at http://localhost:8000
echo API docs will be available at http://localhost:8000/docs
echo.

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
