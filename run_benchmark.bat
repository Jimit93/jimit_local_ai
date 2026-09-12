@echo off
REM Full benchmark: LLM-only baseline vs harness. Takes several minutes on CPU.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0
python scripts\run_benchmark.py --model small %*
