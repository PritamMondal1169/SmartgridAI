@echo off
setlocal enabledelayedexpansion

title SmartgridAI Deployment Launcher

echo ===================================================
echo       SmartgridAI - Full-Stack Deployment
echo ===================================================
echo.

REM Ensure Node.js is in PATH for this session
set "PATH=C:\Program Files\nodejs;%PATH%"

REM Change to project root directory
cd /d "%~dp0"

REM Activate Python Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating Python virtual environment...
    call ".venv\Scripts\activate.bat"
) else (
    echo [!] Warning: .venv not found. Using system python.
)

REM Build frontend if dist doesn't exist
if not exist "frontend\dist\index.html" (
    echo [*] Building React dashboard frontend bundle...
    cd frontend
    call npm.cmd run build
    cd ..
)

REM Check if port 8000 is already in use
netstat -ano | findstr /R /C:":8000 " >nul
if %errorlevel% equ 0 (
    echo [!] Port 8000 is currently occupied.
    echo [*] Binding to Port 8080 instead.
    set "APP_PORT=8080"
) else (
    set "APP_PORT=8000"
)

echo.
echo [*] Starting SmartgridAI Unified Server...
echo     Serving API & Dashboard on http://127.0.0.1:!APP_PORT!
echo.

REM Open browser automatically
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:!APP_PORT!"

REM Start uvicorn
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port !APP_PORT! --reload

pause
