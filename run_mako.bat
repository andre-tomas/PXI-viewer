@echo off
rem Starts the Mako camera GUI with the project's Python environment.
"%~dp0.venv\Scripts\python.exe" "%~dp0Mako_v14.py"
if errorlevel 1 pause
