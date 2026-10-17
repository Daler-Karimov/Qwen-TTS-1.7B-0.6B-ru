@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title Qwen3-TTS WebUI
cd /d "%~dp0"

echo ===================================================
echo             Starting Qwen3-TTS WebUI
echo ===================================================

if exist venv\Scripts\python.exe goto :RUN

echo [INFO] Virtual environment not found. Creating venv...
python -m venv venv
if errorlevel 1 goto :VENV_ERROR

echo [INFO] Updating pip...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet

echo [INFO] Installing dependencies from requirements.txt...
venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 goto :PIP_ERROR

:RUN
echo [INFO] Using virtual environment: venv
venv\Scripts\python.exe webui.py %*
if errorlevel 1 goto :RUN_ERROR
goto :END

:VENV_ERROR
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1

:PIP_ERROR
echo [ERROR] Failed to install dependencies from requirements.txt.
pause
exit /b 1

:RUN_ERROR
echo.
echo [ERROR] An error occurred while launching WebUI.
pause
exit /b 1

:END
pause


