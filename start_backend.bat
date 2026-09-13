@echo off
REM Local AI backend - binds to 127.0.0.1 only.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0
REM Hard CPU cap for the server process tree (cap - 5 = 65%% of all cores).
if not defined JLA_SELF_CPU_CAP set JLA_SELF_CPU_CAP=1
python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8077
