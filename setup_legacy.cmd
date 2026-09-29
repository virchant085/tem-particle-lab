@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv
if errorlevel 1 (
  echo Please install Python 3.12 with the Python launcher, then run setup.cmd again.
  pause
  exit /b 1
)
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. The error above identifies the missing dependency.
  pause
  exit /b 1
)
echo Setup complete. Open start_tracker.cmd to launch the tracker.
pause
