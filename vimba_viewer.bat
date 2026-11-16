@echo off
rem Starts Vimba X Viewer with only Allied Vision's own transport layers.
rem Without this the Viewer also loads the Daheng (Galaxy) and Spinnaker
rem GenTL producers and opens the Mako through Daheng's, where full access fails.
set "GENICAM_GENTL64_PATH=C:\Program Files\Allied Vision\Vimba X\cti"
start "" "C:\Program Files\Allied Vision\Vimba X\bin\VimbaXViewer.exe"
