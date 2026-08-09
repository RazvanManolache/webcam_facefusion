@echo off
setlocal

cd /d "%~dp0"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONUTF8=1"

if not exist "%~dp0webui\dist\index.html" (
    echo Build the React interface once with start_gui.bat before using tray mode.
    pause
    exit /b 1
)

where pythonw.exe >nul 2>nul
if errorlevel 1 (
    echo pythonw.exe was not found. Install the project requirements first.
    pause
    exit /b 1
)

start "FaceFlow Tray" /b pythonw.exe "%~dp0tray_app.py" --port 7862 --enable-virtual-camera
endlocal
