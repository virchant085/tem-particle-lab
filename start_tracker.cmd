@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "TEM_PY=%~dp0.venv-ml\Scripts\pythonw.exe"
if not exist "%TEM_PY%" (
  echo Please run setup_ml.cmd once to install the learning environment.
  pause
  exit /b 1
)
start "" "%TEM_PY%" "%~dp0scripts\launch_tracker.py"
