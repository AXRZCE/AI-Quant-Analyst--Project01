@echo off
echo ===== Starting Project01 Quant Analyst (Local Mode) =====

echo.
echo 1. Starting Backend API...
start cmd /k "cd backend\app && python -m uvicorn simple_app:app --reload --host 0.0.0.0 --port 8000"

echo.
echo 2. Opening Simple UI...
timeout /t 3 > nul
start "" "file://%CD%/simple_ui.html"

echo.
echo ===== Setup Complete =====
echo.
echo Backend API: http://localhost:8000/api
echo Simple UI: simple_ui.html (opened in browser)
echo.
echo Press any key to exit...
pause > nul
