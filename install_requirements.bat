@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

REM Use the python on PATH. Optionally, you can hardcode a venv path here.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Optional: suggest DirectML for Windows GPU
echo.
echo If you have a compatible AMD/Intel/NVIDIA GPU on Windows 10/11, you may install DirectML provider:
echo    pip install onnxruntime-directml
echo.

echo Done.
pause

endlocal
