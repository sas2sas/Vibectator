@echo off
rem Запуск диктофона без окна консоли.
setlocal
set "DIR=%~dp0"
start "" "%DIR%.venv-gigaam\Scripts\pythonw.exe" "%DIR%dictate.py"
