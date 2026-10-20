@echo off
rem Double-click this file to open the Pal Studio. The first run installs what it needs (a few minutes).
cd /d "%~dp0"
python run.py studio
if errorlevel 1 pause
