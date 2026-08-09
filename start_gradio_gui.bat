@echo off
setlocal
cd /d "%~dp0"
set "GRADIO_ANALYTICS_ENABLED=false"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONUTF8=1"
python "%~dp0webcam_deep_swap.py" --gui %*
if errorlevel 1 (
    echo.
    echo Gradio fallback exited with errorlevel %errorlevel%.
    pause
)
endlocal
