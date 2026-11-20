@echo off
title Rephrase AI
cd /d "%~dp0"
echo ==========================================
echo             REPHRASE AI
echo ==========================================
echo.
where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama was not found. Install it from https://ollama.com/download
    pause
    exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer, then try again.
    pause
    exit /b 1
)
python -c "import streamlit, ollama" >nul 2>nul
if errorlevel 1 (
    echo Required packages are missing. Run: pip install -r requirements.txt
    pause
    exit /b 1
)
echo Installed Ollama models:
ollama list
echo.
echo Starting Streamlit...
python -m streamlit run app.py
pause
