@echo off
REM Fix Chumpy Installation (Windows)
REM Installs the correct patched version for getargs compatibility

echo ========================================
echo Fixing Chumpy Installation
echo ========================================
echo.
echo Current commit: 580566ea...
echo Required commit: 9b045ff5...
echo.
echo This fixes: 'getargs() got an unexpected keyword argument' error
echo.

set /p continue="Continue? (y/n): "
if /i not "%continue%"=="y" (
    echo Cancelled.
    exit /b 1
)

echo.
echo Uninstalling current chumpy...
pip uninstall chumpy -y

echo.
echo Installing patched chumpy from git...
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17

echo.
echo ========================================
echo Verifying installation...
echo ========================================
python verify_installation.py

echo.
echo Done!
pause
