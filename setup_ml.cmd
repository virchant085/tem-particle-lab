@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "TEM_TORCH_INDEX=https://download.pytorch.org/whl/cu130"
if /I "%~1"=="cpu" set "TEM_TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if not exist ".venv-ml\Scripts\python.exe" (
  py -3.12 -m venv .venv-ml
  if errorlevel 1 goto :error
)
".venv-ml\Scripts\python.exe" -X utf8 -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv-ml\Scripts\python.exe" -X utf8 -m pip install torch==2.14.0 torchvision==0.29.0 --index-url "%TEM_TORCH_INDEX%"
if errorlevel 1 goto :error
".venv-ml\Scripts\python.exe" -X utf8 -m pip install -r requirements-ml.txt
if errorlevel 1 goto :error
".venv-ml\Scripts\python.exe" -X utf8 scripts\check_ml_environment.py
if errorlevel 1 goto :error
echo Installation complete. Open start_tracker.cmd to start.
pause
exit /b 0
:error
echo Setup failed. Python 3.12 is required. For CPU-only installation, use setup_ml.cmd cpu.
pause
exit /b 1
