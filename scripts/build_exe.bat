@echo off
setlocal

set TARGET=%1
if "%TARGET%"=="" set TARGET=all

powershell -ExecutionPolicy Bypass -File "%~dp0build_exe.ps1" -Target %TARGET%
if errorlevel 1 (
  echo.
  echo ERROR: Сборка завершилась с ошибкой.
  exit /b 1
)

echo.
echo Сборка завершена.
exit /b 0

