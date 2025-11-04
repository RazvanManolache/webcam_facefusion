@echo off
setlocal

REM Change to the directory of this script
cd /d "%~dp0"

REM Optional: disable telemetry
set "GRADIO_ANALYTICS_ENABLED=false"
set "HF_HUB_DISABLE_TELEMETRY=1"

REM Optional: better Unicode handling on Windows
set "PYTHONUTF8=1"

REM Launch the GUI (pass through any extra args)
python "%~dp0webcam_deep_swap.py" --gui %*

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Program exited with errorlevel %errorlevel%.
    echo Press any key to close...
    pause >nul
)

endlocal
