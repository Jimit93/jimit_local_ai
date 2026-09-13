@echo off
REM UI on 5174 proxied to the test backend on 8078.
cd /d "%~dp0..\frontend"
if not defined JLA_UI_PORT set JLA_UI_PORT=5174
if not defined JLA_API_TARGET set JLA_API_TARGET=http://127.0.0.1:8078
node node_modules\vite\bin\vite.js
