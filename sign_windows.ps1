<#
.SYNOPSIS
    Sign AI Capability Server executables for Windows.

.DESCRIPTION
    Windows equivalent of sign.sh (macOS). Uses signtool.exe to sign all .exe and
    .dll files in the PyInstaller dist folder so that Windows Smart App Control /
    SmartScreen does not block them at runtime.

    For local testing, use the self-signed certificate created by
    scripts/setup_msix_cert.ps1 (in the Flutter project).

    For Store submissions, skip signing — Microsoft signs the MSIX during
    Store certification.

.PARAMETER CertPath
    Path to the .pfx code-signing certificate.

.PARAMETER CertPassword
    Password for the .pfx certificate.

.PARAMETER DistDir
    Path to the PyInstaller dist folder to sign.
    Defaults to dist\visarc_ai_server relative to the script location.

.PARAMETER TimestampServer
    RFC 3161 timestamp server URL. Defaults to DigiCert.

.EXAMPLE
    .\sign_windows.ps1
    .\sign_windows.ps1 -CertPath "C:\certs\my_cert.pfx" -CertPassword "secret"
#>

param(
    [string]$CertPath,
    [string]$CertPassword = "VisArcTest123!",
    [string]$DistDir,
    [string]$TimestampServer = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Continue"

# Resolve defaults
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $DistDir) {
    $DistDir = Join-Path $ScriptDir "dist\visarc_ai_server"
}

if (-not $CertPath) {
    # Look for the Flutter project's self-signed cert
    $CertPath = Join-Path $ScriptDir "..\data_storage_pc\certs\visarc_test.pfx"
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  AI Capability Server - Windows Code Signing" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ─── Validate inputs ─────────────────────────────────────────────────────────

if (-not (Test-Path $DistDir)) {
    Write-Host "ERROR: Dist directory not found: $DistDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $CertPath)) {
    Write-Host "ERROR: Certificate not found: $CertPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "To create a self-signed certificate for local testing, run:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File data_storage_pc\scripts\setup_msix_cert.ps1" -ForegroundColor Cyan
    exit 1
}

# Locate signtool.exe from Windows SDK
$signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Recurse -Filter "signtool.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "x64" } |
    Sort-Object { $_.FullName } -Descending |
    Select-Object -First 1

if (-not $signtool) {
    Write-Host "ERROR: signtool.exe not found. Install the Windows SDK." -ForegroundColor Red
    exit 1
}

Write-Host "  signtool:  $($signtool.FullName)" -ForegroundColor White
Write-Host "  cert:      $CertPath" -ForegroundColor White
Write-Host "  dist:      $DistDir" -ForegroundColor White
Write-Host ""

# ─── Collect files to sign ────────────────────────────────────────────────────

$exeFiles = Get-ChildItem -Path $DistDir -Filter "*.exe" -Recurse -File
$dllFiles = Get-ChildItem -Path $DistDir -Filter "*.dll" -Recurse -File
$pydFiles = Get-ChildItem -Path $DistDir -Filter "*.pyd" -Recurse -File
$allFiles = @($exeFiles) + @($dllFiles) + @($pydFiles) | Where-Object { $_ -ne $null }

Write-Host "Files to sign:" -ForegroundColor Cyan
Write-Host "  .exe: $($exeFiles.Count)" -ForegroundColor White
Write-Host "  .dll: $($dllFiles.Count)" -ForegroundColor White
Write-Host "  .pyd: $($pydFiles.Count)" -ForegroundColor White
Write-Host "  Total: $($allFiles.Count)" -ForegroundColor White
Write-Host ""

if ($allFiles.Count -eq 0) {
    Write-Host "WARNING: No signable files found in $DistDir" -ForegroundColor Yellow
    exit 0
}

# ─── Sign files ───────────────────────────────────────────────────────────────
# WDAC/Smart App Control blocks ALL unsigned executables and DLLs, including
# numpy .pyd modules, scipy DLLs, etc. Sign everything unconditionally.
# Batch files into single signtool calls for performance (shared timestamp).

Write-Host "Signing files..." -ForegroundColor Yellow
$signed = 0
$failed = 0
$failedFiles = @()

# Batch files into groups for faster signing (signtool accepts multiple files)
$batchSize = 50
for ($i = 0; $i -lt $allFiles.Count; $i += $batchSize) {
    $batch = $allFiles[$i..[math]::Min($i + $batchSize - 1, $allFiles.Count - 1)]
    $filePaths = $batch | ForEach-Object { $_.FullName }
    
    $result = & $signtool.FullName sign /f $CertPath /p $CertPassword /fd SHA256 /tr $TimestampServer /td SHA256 $filePaths 2>&1
    if ($LASTEXITCODE -eq 0) {
        $signed += $filePaths.Count
    } else {
        # Retry individually to identify which files failed
        foreach ($file in $batch) {
            $result = & $signtool.FullName sign /f $CertPath /p $CertPassword /fd SHA256 /tr $TimestampServer /td SHA256 $file.FullName 2>&1
            if ($LASTEXITCODE -eq 0) {
                $signed++
            } else {
                $failed++
                $failedFiles += $file.FullName
            }
        }
    }
    
    # Progress indicator
    $pct = [math]::Min(100, [math]::Round(($i + $batch.Count) / $allFiles.Count * 100))
    Write-Host "`r  Progress: $pct% ($($signed + $failed) / $($allFiles.Count))" -NoNewline -ForegroundColor DarkGray
}
Write-Host ""

Write-Host ""

# ─── Verification ─────────────────────────────────────────────────────────────

Write-Host "Verifying signatures..." -ForegroundColor Yellow

$verified = 0
$verifyFailed = 0

# Verify the main executable specifically
$mainExe = Join-Path $DistDir "visarc_ai_server.exe"
if (Test-Path $mainExe) {
    $result = & $signtool.FullName verify /pa $mainExe 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  visarc_ai_server.exe: Verified" -ForegroundColor Green
        $verified++
    } else {
        Write-Host "  visarc_ai_server.exe: VERIFICATION FAILED" -ForegroundColor Red
        $verifyFailed++
    }
}

# Spot-check a few other executables
$spotCheck = $exeFiles | Where-Object { $_.Name -ne "visarc_ai_server.exe" } | Select-Object -First 3
foreach ($file in $spotCheck) {
    $result = & $signtool.FullName verify /pa $file.FullName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $($file.Name): Verified" -ForegroundColor Green
        $verified++
    } else {
        Write-Host "  $($file.Name): VERIFICATION FAILED" -ForegroundColor Red
        $verifyFailed++
    }
}

# ─── Summary ──────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Signing Summary" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Signed:    $signed / $($allFiles.Count)" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "  Failed:    $failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
Write-Host "  Verified:  $verified (spot-checked)" -ForegroundColor Green

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "  Failed files:" -ForegroundColor Red
    foreach ($f in $failedFiles) {
        Write-Host "    - $f" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "WARNING: Some files failed to sign. The AI server may still be" -ForegroundColor Yellow
    Write-Host "blocked by Smart App Control / SmartScreen." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "All files signed successfully!" -ForegroundColor Green
Write-Host ""
