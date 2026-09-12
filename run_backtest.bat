@echo off
REM Historical style back-test on held-out posts.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0
python -c "import json;from backend.benchmarks.linkedin_backtest import run_backtest;r=run_backtest();print(json.dumps(r['aggregate'],indent=2))"
