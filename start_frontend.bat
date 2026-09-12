@echo off
cd /d "%~dp0frontend"
if not exist node_modules ( npm install --no-audit --no-fund )
npm run dev
