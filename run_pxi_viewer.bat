@echo off
rem Starts the PXI-viewer with the project's Python environment, scanning data_camera.
"%~dp0.venv\Scripts\python.exe" "%~dp0pxi-viewer.py" "%~dp0data_camera"
if errorlevel 1 pause
