param(
    [ValidateSet("all", "generator", "microscope")]
    [string]$Target = "all",
    [switch]$Clean,
    [switch]$DryRun,
    [int]$MaxPackageSizeMB = 2500
)

$ErrorActionPreference = "Stop"

function Invoke-Step([string]$Title, [scriptblock]$Action) {
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Action
}

function Assert-FileExists([string]$Path, [string]$Message) {
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw $Message
    }
}

function Get-DirSizeMB([string]$Path) {
    if (-not (Test-Path $Path -PathType Container)) {
        return 0.0
    }
    $bytes = (Get-ChildItem $Path -Recurse -File | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) { return 0.0 }
    return [math]::Round($bytes / 1MB, 1)
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

Invoke-Step "Check PyInstaller" {
    @'
import importlib.util
import sys
if importlib.util.find_spec("PyInstaller") is None:
    print("PyInstaller is not installed. Install with: pip install pyinstaller")
    sys.exit(2)
print("PyInstaller is installed")
'@ | python -
}

if ($Clean) {
    Invoke-Step "Clean build/dist" {
        if (Test-Path "build") { Remove-Item "build" -Recurse -Force }
        if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force }
    }
}

$specs = @()
switch ($Target) {
    "all" {
        $specs += "packaging/pyinstaller_generator_v3.spec"
        $specs += "packaging/pyinstaller_microscope_v2.spec"
    }
    "generator" {
        $specs += "packaging/pyinstaller_generator_v3.spec"
    }
    "microscope" {
        $specs += "packaging/pyinstaller_microscope_v2.spec"
    }
}

foreach ($spec in $specs) {
    Assert-FileExists $spec "Missing PyInstaller spec: $spec"
    if ($DryRun) {
        Write-Host "[DryRun] python -m PyInstaller --noconfirm --clean $spec" -ForegroundColor Yellow
        continue
    }
    Invoke-Step "Build $spec" {
        python -m PyInstaller --noconfirm --clean $spec
    }

    $name = [System.IO.Path]::GetFileNameWithoutExtension($spec) -replace '^pyinstaller_', ''
    $artifact = switch ($name) {
        "generator_v3" { "dist/MetallographyGeneratorV3/MetallographyGeneratorV3.exe" }
        "microscope_v2" { "dist/VirtualMicroscopeV2/VirtualMicroscopeV2.exe" }
        default { "" }
    }
    $packageDir = switch ($name) {
        "generator_v3" { "dist/MetallographyGeneratorV3" }
        "microscope_v2" { "dist/VirtualMicroscopeV2" }
        default { "" }
    }
    if (-not [string]::IsNullOrWhiteSpace($artifact)) {
        Assert-FileExists $artifact "Build finished but EXE not found: $artifact"
        $sizeMB = Get-DirSizeMB $packageDir
        Write-Host ("Artifact size: {0} MB | budget: {1} MB" -f $sizeMB, $MaxPackageSizeMB) -ForegroundColor Yellow
        if ($sizeMB -gt $MaxPackageSizeMB) {
            throw ("Artifact {0} exceeds size budget: {1} MB > {2} MB" -f $packageDir, $sizeMB, $MaxPackageSizeMB)
        }
    }
}

if (-not $DryRun) {
    Invoke-Step "Done" {
        Write-Host "EXE packages are built under dist:" -ForegroundColor Green
        Get-ChildItem "dist" | Select-Object Name, LastWriteTime
    }
}
