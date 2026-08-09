@echo off
setlocal

cd /d "%~dp0"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONUTF8=1"

if not exist "%~dp0webui\dist\index.html" (
    echo Building React interface for the first time...
    pushd "%~dp0webui"
    call npm install
    if errorlevel 1 goto :build_error
    call npm run build
    if errorlevel 1 goto :build_error
    popd
)

start "" "http://127.0.0.1:7862/"
python "%~dp0web_api.py" --port 7862 %*

if errorlevel 1 (
    echo.
    echo Program exited with errorlevel %errorlevel%.
    echo Press any key to close...
    pause >nul
)

endlocal
exit /b

:build_error
popd
echo.
echo React build failed. Check that Node.js and npm are installed.
pause
exit /b 1
