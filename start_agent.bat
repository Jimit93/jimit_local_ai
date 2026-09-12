@echo off
REM Starts backend and UI together, then opens the browser.
cd /d "%~dp0"
start "Jimit Local AI - backend" cmd /k "%~dp0start_backend.bat"
timeout /t 3 /nobreak >nul
start "Jimit Local AI - frontend" cmd /k "%~dp0start_frontend.bat"
timeout /t 4 /nobreak >nul
start http://127.0.0.1:5173
