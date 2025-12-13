# AI Capability Server - Build Script (Windows PowerShell)
# This script builds the standalone executable using PyInstaller

# Stop on errors
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Blue
Write-Host "AI Capability Server - Build Script (Windows)" -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated" -ForegroundColor Yellow
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    } else {
        Write-Host "❌ ERROR: Virtual environment not found at venv\" -ForegroundColor Red
        Write-Host "   Please create it first: python -m venv venv" -ForegroundColor Red
        exit 1
    }
}

# Verify PyInstaller is installed
Write-Host "Checking for PyInstaller..." -ForegroundColor Cyan
try {
    $pyinstallerVersion = & pyinstaller --version 2>&1
    Write-Host "✅ PyInstaller installed: $pyinstallerVersion" -ForegroundColor Green
} catch {
    Write-Host "PyInstaller not found. Installing..." -ForegroundColor Yellow
    & pip install pyinstaller
}

# Check for required directories
Write-Host ""
Write-Host "Checking required directories..." -ForegroundColor Cyan

if (-not (Test-Path "binary")) {
    Write-Host "❌ ERROR: binary\ directory not found" -ForegroundColor Red
    Write-Host "   Please create it and add llama binaries (.exe files)" -ForegroundColor Red
    Write-Host "   Required: llama-server.exe, llama-cli.exe, llama-mtmd-cli.exe, llama-embedding.exe" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "model")) {
    Write-Host "❌ ERROR: model\ directory not found" -ForegroundColor Red
    Write-Host "   Please create it and add model files (.gguf)" -ForegroundColor Red
    exit 1
}

# Count files
$binaryCount = (Get-ChildItem -Path "binary" -File).Count
$modelCount = (Get-ChildItem -Path "model" -Filter "*.gguf" -File).Count

Write-Host "✅ Found $binaryCount binaries in binary\" -ForegroundColor Green
Write-Host "✅ Found $modelCount model files in model\" -ForegroundColor Green

# Check if we have .exe files
$exeCount = (Get-ChildItem -Path "binary" -Filter "*.exe" -File).Count
if ($exeCount -eq 0) {
    Write-Host ""
    Write-Host "⚠️  WARNING: No .exe files found in binary\" -ForegroundColor Yellow
    Write-Host "   Windows requires .exe binaries (llama-server.exe, llama-cli.exe, etc.)" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Build cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Check directory sizes
Write-Host ""
Write-Host "Checking directory sizes..." -ForegroundColor Cyan
$modelSize = (Get-ChildItem -Path "model" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
$binarySize = (Get-ChildItem -Path "binary" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("  Model directory: {0:N2} MB" -f $modelSize) -ForegroundColor White
Write-Host ("  Binary directory: {0:N2} MB" -f $binarySize) -ForegroundColor White

Write-Host ""
Write-Host "Building AI Capability Server executable..." -ForegroundColor Cyan
Write-Host ""

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Cyan
if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
}
if (Test-Path "dist") {
    Remove-Item -Path "dist" -Recurse -Force
}

# Build with PyInstaller
Write-Host "Running PyInstaller..." -ForegroundColor Cyan
Write-Host ""

try {
    & pyinstaller --clean ai_capability.spec
    $buildSuccess = $LASTEXITCODE -eq 0
} catch {
    $buildSuccess = $false
}

# Check if build was successful
if ($buildSuccess -and (Test-Path "dist\ai_capability_server\ai_capability_server.exe")) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output location: dist\ai_capability_server\" -ForegroundColor White
    Write-Host ""
    Write-Host "To run the application:" -ForegroundColor Cyan
    Write-Host "  cd dist\ai_capability_server" -ForegroundColor Yellow
    Write-Host "  .\ai_capability_server.exe" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To distribute:" -ForegroundColor Cyan
    Write-Host "  cd dist" -ForegroundColor Yellow
    Write-Host "  Compress-Archive -Path ai_capability_server -DestinationPath ai_capability_server.zip" -ForegroundColor Yellow
    Write-Host ""
    
    # Show size
    $distSize = (Get-ChildItem -Path "dist\ai_capability_server" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host ("Total packaged size: {0:N2} MB" -f $distSize) -ForegroundColor White
    Write-Host ""
    
    Write-Host "Note: The executable includes all dependencies and models." -ForegroundColor Cyan
    Write-Host "      It can be distributed as a single folder." -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "❌ Build failed!" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Cyan
    Write-Host "  - Missing .exe binaries in binary\ directory" -ForegroundColor White
    Write-Host "  - Missing model files in model\ directory" -ForegroundColor White
    Write-Host "  - PyInstaller not properly installed" -ForegroundColor White
    Write-Host "  - Insufficient disk space" -ForegroundColor White
    Write-Host ""
    exit 1
}
