@echo off
echo === CSUSM Campus Monitor ===

REM Check ffmpeg
where ffmpeg >nul 2>&1 || (echo ERROR: ffmpeg not found on PATH. Install it first. && pause && exit /b 1)

REM Create venv if needed
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and install deps
call venv\Scripts\activate.bat
pip install -q -r backend\requirements.txt

REM Start server and open browser
echo Starting server at http://localhost:8000 ...
start http://localhost:8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
