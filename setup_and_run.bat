@echo off
REM === Step 1: Set up virtual environment ===
SET VENV_DIR=.venv
IF NOT EXIST %VENV_DIR% (
    python -m venv %VENV_DIR%
)

REM === Step 2: Activate virtual environment ===
CALL %VENV_DIR%\Scripts\activate.bat

REM === Step 3: Install requirements ===
pip install --upgrade pip
pip install -r requirements.txt

REM === Step 4: Set model_dir in main.py to current directory ===
SETLOCAL ENABLEDELAYEDEXPANSION
SET CURR_DIR=%CD%\models
REM Escape backslashes for Python string
SET ESCAPED_DIR=!CURR_DIR:\=\\!
REM Use PowerShell to replace the line in main.py
powershell -Command "(Get-Content main.py) -replace 'model_dir = r\".*?\"', 'model_dir = r"!ESCAPED_DIR!"' | Set-Content main.py"
ENDLOCAL

REM === Step 5: Start FastAPI server ===
echo Starting FastAPI server...
uvicorn main:app --host 0.0.0.0 --port 8000 