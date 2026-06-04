@echo off
cd /d "%~dp0"
echo Updating Fizgig...
git pull
echo.
echo Installing/updating dependencies...
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
    echo WARNING: venv not found - run install_fizgig.bat to set it up.
)
echo.
echo Update complete! Run Fizgig with run_fizgig.bat
pause
