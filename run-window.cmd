@echo off
rem Window mode: a button instead of a hotkey (works without keyboard hooks).
setlocal
set "DIR=%~dp0"
set "PYTHONUTF8=1"
start "" "%DIR%.venv-gigaam\Scripts\pythonw.exe" "%DIR%dictate.py" --window
