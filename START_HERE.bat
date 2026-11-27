@echo off
rem Double-click to open the project website with live target.exe simulations.
rem First run: sets up Python packages (needs internet once). See README_FOR_SIR.md.
title Electron gun ANN - local server
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if defined PY goto run

where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Python 3.10 or newer is needed: https://www.python.org/downloads/
  echo Tick "Add python.exe to PATH" when installing, then double-click this file again.
  pause
  exit /b 1
)

%PY% -c "import numpy, pandas, mpmath" >nul 2>&1
if not errorlevel 1 goto run

echo First run: installing the Python packages into .venv (one time, needs internet) ...
%PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-local.txt || (echo pip install failed & pause & exit /b 1)
set "PY=.venv\Scripts\python.exe"

:run
%PY% serve_local.py %*
pause
