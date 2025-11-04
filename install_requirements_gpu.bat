@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

REM Upgrade pip
python -m pip install --upgrade pip

REM Ensure CPU ONNX Runtime is not installed to avoid conflicts
python - <<PYCODE
import sys, subprocess
try:
    import onnxruntime  # CPU package name
    print("Uninstalling conflicting 'onnxruntime' (CPU) package...")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"]) 
except Exception:
    pass
PYCODE

REM Install GPU requirements (onnxruntime-gpu)
python -m pip install -r requirements-gpu.txt

REM Quick note about NVIDIA drivers
echo.
echo NOTE:
echo  - onnxruntime-gpu uses CUDA/cuDNN wheels; you only need a recent NVIDIA driver.
echo  - Do NOT install the plain 'onnxruntime' CPU wheel together with onnxruntime-gpu.
echo.

echo Done.
pause

endlocal
