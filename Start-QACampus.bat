 @echo off
setlocal
cd /d "%~dp0"
echo Starting QA Campus at http://localhost:5110
echo.
dotnet run --project Fyp.csproj --launch-profile QACampus
pause
