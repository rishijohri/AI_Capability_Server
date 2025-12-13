@echo off
REM AI Capability Server - Build Script (Windows Batch)
REM This script builds the standalone executable using PyInstaller

setlocal enabledelayedexpansion

echo ============================================================
echo AI Capability Server - Build Script (Windows)
echo ============================================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo [WARNING] Virtual environment not activated
    echo Activating virtual environment...
    
    if exist "venv\Scripts\activate.bat" (
        call venv\Scripts\activate.bat
    ) else (
        echo [ERROR] Virtual environment not found at venv\
        echo Please create it first: python -m venv venv
        exit /b 1
    )
)

REM Verify PyInstaller is installed
echo Checking for PyInstaller...
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Check for required directories
echo.
echo Checking required directories...

if not exist "binary" (
    echo [ERROR] binary\ directory not found
    echo Please create it and add llama binaries ^(.exe files^)
    echo Required: llama-server.exe, llama-cli.exe, llama-mtmd-cli.exe, llama-embedding.exe
    exit /b 1
)

if not exist "model" (
    echo [ERROR] model\ directory not found
    echo Please create it and add model files ^(.gguf^)
    exit /b 1
)

REM Count files
set binary_count=0
set model_count=0
set exe_count=0

for %%f in (binary\*) do set /a binary_count+=1
for %%f in (model\*.gguf) do set /a model_count+=1
for %%f in (binary\*.exe) do set /a exe_count+=1

echo [OK] Found !binary_count! binaries in binary\
echo [OK] Found !model_count! model files in model\

if !exe_count! equ 0 (
    echo.
    echo [WARNING] No .exe files found in binary\
    echo Windows requires .exe binaries ^(llama-server.exe, llama-cli.exe, etc.^)
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" (
        echo Build cancelled.
        exit /b 0
    )
)

echo.
echo Building AI Capability Server executable...
echo.

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM Build with PyInstaller
echo Running PyInstaller...
echo.

pyinstaller --clean ai_capability.spec

REM Check if build was successful
if exist "dist\ai_capability_server\ai_capability_server.exe" (
    echo.
    echo ============================================================
    echo [SUCCESS] Build successful!
    echo ============================================================
    echo.
    echo Output location: dist\ai_capability_server\
    echo.
    echo To run the application:
    echo   cd dist\ai_capability_server
    echo   ai_capability_server.exe
    echo.
    echo To distribute:
    echo   cd dist
    echo   tar -a -c -f ai_capability_server.zip ai_capability_server
    echo   ^(or use Windows Explorer to create a ZIP file^)
    echo.
) else (
    echo.
    echo ============================================================
    echo [ERROR] Build failed!
    echo ============================================================
    echo.
    echo Please check the error messages above.
    echo.
    echo Common issues:
    echo   - Missing .exe binaries in binary\ directory
    echo   - Missing model files in model\ directory
    echo   - PyInstaller not properly installed
    echo   - Insufficient disk space
    echo.
    exit /b 1
)

endlocal
