@echo off
setlocal

REM Change to the directory of this script
cd /d "%~dp0"

REM Add the FaceFusion subfolder to PYTHONPATH so `python -m facefusion` works
set "PYTHONPATH=%~dp0facefusion_mrg;%PYTHONPATH%"

REM Optional: disable telemetry
set "GRADIO_ANALYTICS_ENABLED=false"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONUTF8=1"

REM Launch FaceFusion UI (pass through any extra args)
python -m facefusion --gui %*

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo FaceFusion exited with errorlevel %errorlevel%.
    echo Press any key to close...
    pause >nul
)

endlocal
