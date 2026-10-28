@echo off
rem Builds a standalone Windows app (no Python required).
rem Run once: needs .venv-build with onnxruntime, pyinstaller and friends.
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv-build\Scripts\python.exe"

if not exist "%PY%" (
  echo Build environment not found: %PY%
  pause
  exit /b 1
)

echo === icon ===
"%PY%" make-icon.py

echo === pyinstaller ===
"%PY%" -m PyInstaller --noconfirm --clean "%~dp0Vibectator.spec"
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo.
echo Done: dist\Vibectator\Vibectator.exe
pause
