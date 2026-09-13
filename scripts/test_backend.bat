@echo off
REM Second backend for UI/E2E testing beside the everyday one on 8077. No model warm-up.
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%cd%
set JLA_SKIP_WARMUP=1
if not defined JLA_TEST_PORT set JLA_TEST_PORT=8078
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port %JLA_TEST_PORT%
