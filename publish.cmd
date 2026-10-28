@echo off
rem One-click publish: stage everything, commit with a timestamp, push to GitHub.
rem Usage: just double-click this file after you changed something.
setlocal
cd /d "%~dp0"

echo === git status ===
git status --short
echo.

git add -A
git commit -m "update %DATE% %TIME%" >nul 2>&1
if errorlevel 1 (
  echo Nothing new to commit - just pushing.
) else (
  echo Committed.
)

echo === pushing ===
git push
if errorlevel 1 (
  echo.
  echo Push failed. Check: git remote -v, and that you are signed in to GitHub.
) else (
  echo.
  echo Done. Changes are on GitHub.
)

echo.
pause
