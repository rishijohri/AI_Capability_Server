@echo off
REM Quick Start: Model Download Feature
REM This script helps you get started with downloading models from Hugging Face

echo ==================================================
echo   AI Capability - Model Download Quick Start
echo ==================================================
echo.

REM Check if running from project root
if not exist "run_server.py" (
    echo Error: Please run this script from the project root directory
    echo Example: tests\quickstart_download.bat
    exit /b 1
)

echo Step 1: Installing dependencies...
echo -----------------------------------
pip install huggingface_hub>=0.20.0
if %errorlevel% neq 0 (
    echo Failed to install huggingface_hub
    exit /b 1
)
echo Dependencies installed
echo.

echo Step 2: Validating configuration...
echo ------------------------------------
python tests\validate_download_config.py
if %errorlevel% neq 0 (
    echo.
    echo Configuration issues found!
    echo.
    echo Please edit app\config\settings.py and add 'repo_id' fields to models.
    echo See REPO_ID_EXAMPLES.md for examples.
    echo.
    echo Example:
    echo   "qwen_3_0.6B": {
    echo       "model_file": "Qwen3-0.6B-Q4_K_M.gguf",
    echo       "name": "qwen_3_0.6B",
    echo       "type": "chat",
    echo       "repo_id": "Qwen/Qwen3-0.6B-GGUF"  # Add this line
    echo   }
    echo.
    echo After adding repo_id values, run this script again.
    exit /b 1
)
echo Configuration validated
echo.

echo Step 3: Starting server (in background)...
echo -------------------------------------------
REM Check if server is already running
netstat -ano | findstr :8000 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo Server already running on port 8000
) else (
    start /B python run_server.py > ai_capability_server.log 2>&1
    echo Waiting for server to start...
    timeout /t 3 /nobreak >nul
    
    REM Check if server started successfully
    netstat -ano | findstr :8000 | findstr LISTENING >nul
    if %errorlevel% neq 0 (
        echo Failed to start server
        echo Check logs at ai_capability_server.log
        exit /b 1
    )
    echo Server started successfully
)
echo.

echo ==================================================
echo   Ready to Download Models!
echo ==================================================
echo.
echo Option 1: Use the test script (Python)
echo   python tests\test_download_models.py
echo.
echo Option 2: Use Python interactive shell
echo   python -c "import asyncio, json, websockets; asyncio.run(download())"
echo   (where download() is defined in test_download_models.py)
echo.
echo Documentation:
echo   - MODEL_DOWNLOAD_GUIDE.md - Complete guide
echo   - REPO_ID_EXAMPLES.md - Configuration examples
echo   - API_REFERENCE.md - API documentation
echo.
echo Recommended first download: qwen_3_0.6B (~600MB)
echo.
pause
