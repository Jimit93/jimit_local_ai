@echo off
REM One-click launcher: starts backend + UI, waits until both actually answer
REM (not a guessed timeout - startup time varies), then opens the dashboard.
cd /d "%~dp0"
title Jimit Local AI - launcher

start "Jimit Local AI - backend" cmd /k "%~dp0start_backend.bat"
start "Jimit Local AI - frontend" cmd /k "%~dp0start_frontend.bat"

echo(
echo Starting Jimit Local AI...
echo   backend  http://127.0.0.1:8077  (the model finishes warming up in the background)
echo   frontend http://127.0.0.1:5173
echo(

set /a _tries=0
:wait_backend
curl -s -o nul -m 2 http://127.0.0.1:8077/api/health
if not errorlevel 1 goto :backend_up
set /a _tries+=1
if %_tries% GEQ 90 goto :backend_up
timeout /t 1 /nobreak >nul
goto :wait_backend
:backend_up

set /a _tries=0
:wait_frontend
curl -s -o nul -m 2 http://127.0.0.1:5173
if not errorlevel 1 goto :open_dashboard
set /a _tries+=1
if %_tries% GEQ 60 goto :open_dashboard
timeout /t 1 /nobreak >nul
goto :wait_frontend

:open_dashboard
echo Opening the dashboard...
start http://127.0.0.1:5173
