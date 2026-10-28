@echo off
rem Stops the running dictation app: asks it to exit, then kills it by PID if needed.
setlocal enabledelayedexpansion
set "DIR=%~dp0"
set "PIDFILE=%DIR%vibectator.pid"

if exist "%PIDFILE%" (
  set /p DPID=<"%PIDFILE%"
  echo Sending stop signal to PID !DPID! ...
  "%DIR%.venv-gigaam\Scripts\python.exe" "%DIR%dictate.py" --stop
  timeout /t 3 <nul >nul
  taskkill /F /PID !DPID! >nul 2>&1
  if exist "%PIDFILE%" del "%PIDFILE%" >nul 2>&1
  echo Done.
) else (
  echo No PID file, sending stop signal anyway...
  "%DIR%.venv-gigaam\Scripts\python.exe" "%DIR%dictate.py" --stop
)
timeout /t 2 <nul >nul

