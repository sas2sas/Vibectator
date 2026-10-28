@echo off
rem Запуск с видимым окном консоли: видно статусы и распознанный текст.
setlocal
set "DIR=%~dp0"
set "PYTHONUTF8=1"
"%DIR%.venv-gigaam\Scripts\python.exe" "%DIR%dictate.py" %*
echo.
echo Программа завершилась. Код: %ERRORLEVEL%
pause
